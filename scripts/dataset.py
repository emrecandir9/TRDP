"""Frozen inputs, protected composition, saved-image verification and human review."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import silhouette

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / 'config/recipe.json'
DATA = ROOT / 'data'
# detached_hand_silhouette is the paper-matched detached hand (hand on black, cut at the
# wrist); detached_hand is the earlier rectangle, kept as a sensitivity condition. The frozen
# sources and the historical accepted archive contain only the first three contexts.
CONTEXTS = ('full_body', 'upper_limb', 'detached_hand', 'detached_hand_silhouette')
HISTORICAL_CONTEXTS = CONTEXTS[:3]
CHECKS = ('natural_table_texture', 'smooth_desk_transition', 'hand_and_contact_preserved',
          'body_and_clothing_preserved', 'paired_action_correct', 'lighting_natural',
          'silhouette_clean_on_black')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def rgb(path):
    with Image.open(path) as image:
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (2048, 2048),
                f'Expected a 2048-square RGB PNG: {path}')
        return np.array(image)


def frozen_files(directory, files):
    for name, expected in files.items():
        path = Path(directory) / name
        require(path.is_file(), f'Missing local file: {path}. Restore the frozen data bundle first.')
        require(sha(path) == expected, f'Changed frozen file: {path}')


def identity():
    return {name: sha(ROOT / name) for name in
            ('config/recipe.json', 'data/frozen_sources.json', 'data/hand_segmentation.json', 'data/silhouettes.json',
             'scripts/dataset.py', 'scripts/generate.py', 'scripts/models.py', 'scripts/segment_hands.py',
             'scripts/silhouette.py')}


def plan():
    cfg = read(RECIPE)
    require(sha(DATA / 'frozen_sources.json') == cfg['frozen_sources_sha256'], 'Changed source manifest')
    require(sha(DATA / 'silhouettes.json') == cfg['silhouettes_sha256'], 'Changed silhouette manifest')
    sources = {s['id']: s for s in read(DATA / 'frozen_sources.json')['sources']}
    jobs = []
    for source in sources.values():
        for desk in cfg['desks']:
            for seed in cfg['seeds']:
                base = f"r{source['reference']:02d}_{desk}_s{seed}"
                for light in cfg['lighting']:
                    name = f'{base}_{light}'
                    jobs.append(dict(id=name, base_id=base, source_id=source['id'],
                                     desk_id=desk, seed=seed, light_id=light,
                                     canary=name in cfg['canary']))
    require(len(sources) == 20 and len(jobs) == len({j['id'] for j in jobs}) == 480,
            'Expected 20 sources and 480 unique families')
    require(sum(j['canary'] for j in jobs) == 10 and len({j['base_id'] for j in jobs}) == 160,
            'Expected 10 canaries and 160 diffusion parents')
    return cfg, sources, jobs


def check_sources(sources):
    for source in sources.values():
        directory = DATA / 'sources' / source['source_directory']
        inputs = DATA / 'inputs' / source['id']
        frozen_files(directory, source['source_condition_sha256'])
        frozen_files(inputs, source['input_sha256'])
        a, b = rgb(directory / 'full_body/static.png'), rgb(directory / 'full_body/movement.png')
        with Image.open(inputs / 'desk_weight.png') as image:
            require(image.mode == 'L' and image.size == (2048, 2048), 'Invalid desk weight')
            weight = np.array(image)
        require(not weight[np.any(a != b, axis=2)].any(), f"Desk overlaps action: {source['id']}")
        check_context_pixels(directory, a, b, source)
        silhouette.check_source(source, a, b)
        for name, mode, size, image_format in (
                ('desk_mask.png', 'L', (2048, 2048), 'PNG'),
                ('image_1024.png', 'RGB', (1024, 1024), 'PNG'),
                ('control_1024.png', 'RGB', (1024, 1024), 'PNG'),
                ('mask_review.jpg', 'RGB', (1024, 1024), 'JPEG')):
            with Image.open(inputs / name) as image:
                image.load()
                require(image.mode == mode and image.size == size and image.format == image_format,
                        f'Invalid frozen input: {inputs / name}')
                if name == 'desk_mask.png':
                    require(np.array_equal(np.array(image), (weight > 0).astype(np.uint8) * 255),
                            'Binary desk mask disagrees with blending weight')


def lighting(image, gains):
    if list(gains) == [1., 1., 1.]:
        return image.copy()
    x = image.astype(np.float32) / 255.
    linear = np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
    linear = np.clip(linear * np.array(gains, np.float32), 0, 1)
    value = np.where(linear <= .0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - .055)
    return np.rint(value * 255).clip(0, 255).astype(np.uint8)


def shared_desk(a, b, proposal, weight, offset_limit=24):
    require(a.shape == b.shape == proposal.shape and weight.shape == a.shape[:2], 'Composition shape mismatch')
    editable = weight > 0
    require(not editable[np.any(a != b, axis=2)].any(), 'Desk mask overlaps accepted action')
    core = weight >= 250
    require(core.sum() > 1000, 'Desk core is too small')
    offset = np.median(a[core].astype(float), axis=0) - np.median(proposal[core].astype(float), axis=0)
    offset = np.clip(offset, -offset_limit, offset_limit)
    corrected = np.clip(proposal.astype(float) + offset, 0, 255)
    alpha = weight.astype(float)[:, :, None] / 255.
    new_a = np.rint(a * (1 - alpha) + corrected * alpha).clip(0, 255).astype(np.uint8)
    new_a[~editable] = a[~editable]
    new_b = b.copy()
    new_b[editable] = new_a[editable]
    require(np.array_equal(new_b[~editable], b[~editable]) and
            np.array_equal(new_a[editable], new_b[editable]), 'Protected composition changed')
    changed = int(np.any(new_a != a, axis=2).sum())
    require(changed > 0, 'No desk variation produced')
    return new_a, new_b, dict(shared_rgb_offset=offset.tolist(), desk_changed_pixels=changed)


def context_mask(shape, source, context):
    keep = np.zeros(shape, bool)
    if context == 'full_body':
        keep[:] = True
    elif context == 'upper_limb':
        keep[:source['upper_limb_cutoff']] = True
    elif context == 'detached_hand':
        x, y, right, bottom = source['used_hand_box']
        keep[y:bottom, x:right] = True
    else:
        raise ValueError(context)
    return keep


def context_pixels(full, source, context, frame):
    """Condition image: retained full-body pixels, black elsewhere; the silhouette is soft-edged."""
    if context == 'detached_hand_silhouette':
        return silhouette.apply(full, silhouette.load(source, frame))
    pixels = np.zeros_like(full)
    keep = context_mask(full.shape[:2], source, context)
    pixels[keep] = full[keep]
    return pixels


def audit_pair(a, b, original_a, original_b, weight, gains):
    protected = weight == 0
    for actual, original in ((a, original_a), (b, original_b)):
        require(np.array_equal(actual[protected], lighting(original, gains)[protected]), 'Protected pixels changed')
    delta = np.any(a != b, axis=2)
    allowed = np.any(original_a != original_b, axis=2)
    require(not delta[~allowed].any() and delta.any(), 'Action changed outside its original support or vanished')


def check_context_pixels(directory, a, b, source, contexts=HISTORICAL_CONTEXTS[1:]):
    for context in contexts:
        for label, full in (('static', a), ('movement', b)):
            require(np.array_equal(rgb(directory / f'{context}/{label}.png'), context_pixels(full, source, context, label)),
                    'Context pixels changed')


def export_family(directory, a, b, original_a, original_b, source, job, weight, gains, provenance):
    audit_pair(a, b, original_a, original_b, weight, gains)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    files = {}
    for context in CONTEXTS:
        (directory / context).mkdir()
        for label, full in (('static', a), ('movement', b)):
            pixels = context_pixels(full, source, context, label)
            name = f'{context}/{label}.png'
            Image.fromarray(pixels).save(directory / name)
            actual = rgb(directory / name)
            require(np.array_equal(actual, pixels), 'Saved export differs from intended pixels')
            files[name] = sha(directory / name)
    record = dict(job, files=files, provenance=provenance, source_lineage_group=source['source_lineage_group'],
                  status='numeric_pass_visual_review_pending', independent_identity_claim=False)
    write(directory / 'audit.json', record)
    return record


def check_family_pixels(directory, source, gains, weight=None, contexts=CONTEXTS[1:]):
    original = DATA / 'sources' / source['source_directory']
    original_a, original_b = rgb(original / 'full_body/static.png'), rgb(original / 'full_body/movement.png')
    a, b = rgb(directory / 'full_body/static.png'), rgb(directory / 'full_body/movement.png')
    if weight is None:
        weight = np.array(Image.open(DATA / 'inputs' / source['id'] / 'desk_weight.png'))
    audit_pair(a, b, original_a, original_b, weight, gains)
    check_context_pixels(directory, a, b, source, contexts)
    return a, b, original_b, weight


def verify_family(output, job, source, cfg):
    directory = output / 'candidates' / job['id']
    record = read(directory / 'audit.json')
    require(all(record[k] == v for k, v in job.items()), f"Candidate identity changed: {job['id']}")
    require(record['provenance']['identity'] == identity(), 'Candidate code or recipe changed')
    require(record['provenance']['run_sha256'] == sha(output / 'provenance.json'), 'Candidate runtime changed')
    parent = output / 'parents' / job['base_id']
    require(record['provenance']['parent_sha256'] == sha(parent / 'result.json'), 'Parent record changed')
    parent_record = read(parent / 'result.json')
    require(parent_record['run_sha256'] == sha(output / 'provenance.json'), 'Parent runtime changed')
    frozen_files(parent, parent_record['files'])
    frozen_files(directory, record['files'])
    expected_files = {f'{c}/{label}.png' for c in CONTEXTS for label in ('static', 'movement')}
    require(set(record['files']) == expected_files, f'Expected exactly {len(expected_files)} condition PNGs')
    gains = cfg['lighting'][job['light_id']]
    a, b, original_b, weight = check_family_pixels(directory, source, gains)
    new_a = rgb(parent / 'static_with_shared_desk.png')
    new_b = original_b.copy()
    new_b[weight > 0] = new_a[weight > 0]
    require(np.array_equal(a, lighting(new_a, gains)) and np.array_equal(b, lighting(new_b, gains)),
            'Candidate differs from its recorded parent')
    return dict(id=job['id'], audit_sha256=sha(directory / 'audit.json'), condition_pngs=2 * len(CONTEXTS))


def verify(output, family=None):
    cfg, sources, jobs = plan()
    check_sources(sources)
    selected = [j for j in jobs if j['id'] == family] if family else [
        j for j in jobs if (output / 'candidates' / j['id']).exists()]
    require(selected, 'No candidates to verify')
    records = [verify_family(output, j, sources[j['source_id']], cfg) for j in selected]
    report = dict(families_checked=len(records), condition_pngs_checked=2 * len(CONTEXTS) * len(records),
                  visual_quality_certified=False, records=records)
    path = output / 'candidates' / family / 'verification.json' if family else output / 'verification.json'
    write(path, report)
    return report


def record_review(output, path):
    supplied = read(path)
    # Only a human decision approves a stimulus; automated checks and assistant inspections are advisory.
    require(supplied['reviewer'] == 'human_user', 'Only a human reviewer can approve stimuli (reviewer: human_user)')
    require(supplied['identity'] == identity(), 'Review applies to changed code or recipe')
    _, sources, jobs = plan()
    check_sources(sources)
    by_id = {j['id']: j for j in jobs}
    ledger_path = output / 'reviews.json'
    ledger = read(ledger_path) if ledger_path.exists() else dict(identity=identity(), cases={})
    require(ledger['identity'] == identity(), 'Existing review belongs to another version')
    require(bool(supplied['cases']), 'No review decisions supplied')
    for name, decision in supplied['cases'].items():
        require(name not in ledger['cases'], 'Reviews are frozen; use a separate output directory for a revised run')
        job = by_id[name]
        verified = verify_family(output, job, sources[job['source_id']], read(RECIPE))
        require(decision['audit_sha256'] == verified['audit_sha256'], 'Candidate changed after review template')
        require(type(decision['passed']) is bool, 'Review needs an explicit boolean decision')
        require(all(decision['checks'].get(k) is True for k in CHECKS) if decision['passed'] else
                bool(decision.get('reason')), 'Passing reviews need all checks; rejection needs a reason')
        ledger['cases'][name] = dict(decision, reviewer=supplied['reviewer'], input_review_sha256=sha(path),
                                     reviewed_at_utc=datetime.now(timezone.utc).isoformat())
    write(ledger_path, ledger)
    accepted = [dict(id=name, **decision) for name, decision in ledger['cases'].items() if decision['passed']]
    write(output / 'accepted.json', dict(identity=identity(), families=accepted,
                                        image_count=2 * len(CONTEXTS) * len(accepted),
                                        scientific_replication_validated=False))


def require_canary(output):
    cfg, sources, jobs = plan()
    ledger = read(output / 'reviews.json')
    require(ledger['identity'] == identity(), 'Canary review belongs to another version')
    for job in jobs:
        if job['canary']:
            decision = ledger['cases'].get(job['id'], {})
            require(decision.get('passed') is True, f"Canary needs visual approval: {job['id']}")
            verified = verify_family(output, job, sources[job['source_id']], cfg)
            require(decision['audit_sha256'] == verified['audit_sha256'], 'Canary changed after visual approval')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check', 'verify', 'review-template', 'review'))
    parser.add_argument('review_file', nargs='?', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'output')
    parser.add_argument('--family')
    args = parser.parse_args()
    if args.command == 'check':
        _, sources, jobs = plan()
        check_sources(sources)
        print('Verified 20 sources, 120 condition PNGs, 100 inputs, 40 silhouette mattes; '
              '480 families / 160 parents / 10 canaries.')
    elif args.command == 'verify':
        report = verify(args.output, args.family)
        print(json.dumps({k: v for k, v in report.items() if k != 'records'}))
    elif args.command == 'review-template':
        report = verify(args.output, args.family)
        ledger = read(args.output / 'reviews.json') if (args.output / 'reviews.json').exists() else {'cases': {}}
        template = dict(reviewer='human_user', identity=identity(), cases={r['id']:
                        dict(audit_sha256=r['audit_sha256'], passed=False, reason='',
                             checks={k: False for k in CHECKS}) for r in report['records'] if r['id'] not in ledger['cases']})
        print(json.dumps(template, indent=2))
    else:
        require(args.review_file is not None, 'Supply a completed review JSON file')
        record_review(args.output, args.review_file)
        print('Saved hash-bound visual decisions. Numeric checks do not certify realism.')
