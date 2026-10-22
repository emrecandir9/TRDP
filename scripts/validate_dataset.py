"""Validate every downloaded original family against historical approval and frozen pixels."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
from dataset import (ROOT, HISTORICAL_CONTEXTS, read, write, sha, require, plan, check_sources,
                     frozen_files, check_family_pixels, rgb, lighting)
import silhouette

MANIFEST = ROOT / 'docs/accepted-dataset.json'


def check_record(directory, family, job, source):
    audit = read(directory / 'audit.json')
    files = family['condition_png_sha256']
    expected = {f'{context}/{label}.png' for context in HISTORICAL_CONTEXTS for label in ('static', 'movement')}
    require(set(files) == expected and audit['files'] == files, 'Image list disagrees with approved manifest')
    require(family['decision'] == 'approve' and family['reviewer'] == 'human_user', 'Missing historical human approval')
    require(audit['id'] == family['id'] == job['id'] and audit['source_id'] == family['source_id'] == source['id'],
            'Family/source identity mismatch')
    require(audit['reference'] == family['reference'] == source['reference'], 'Reference mismatch')
    require(audit['desk_id'] == job['desk_id'] and audit['seed'] == job['seed'] and
            audit['lighting'] == job['light_id'], 'Appearance recipe mismatch')
    require(audit['source_lineage_group'] == family['source_lineage_group'] == source['source_lineage_group'],
            'Source grouping mismatch')
    require(audit['source_condition_sha256'] == source['source_condition_sha256'], 'Frozen source hashes changed')
    require(family['known_clothing_issue_flag'] == (source['reference'] == 17), 'Reference-17 flag changed')
    frozen_files(directory, files)
    return dict(id=family['id'], audit_sha256=sha(directory / 'audit.json'), condition_pngs_checked=6)


def historical_weight(directory, job, source, audit):
    provenance = directory / '_provenance'
    parent = provenance / 'diffusion_parents' / job['base_id'] / 'result.json'
    require(sha(parent) == audit['provenance']['diffusion_parent_result_sha256'], 'Historical parent record changed')
    record = read(parent)
    require(record['job']['id'] == job['base_id'] and record['job']['source_id'] == source['id'],
            'Historical parent identity mismatch')
    expected = dict(source['input_sha256'])
    expected.update(source['mask_coordinate_correction']['previous_input_sha256'])
    require(record['frozen_input_sha256'] == expected, 'Historical input version mismatch')
    path = provenance / 'inputs' / source['id'] / 'desk_weight.png'
    require(sha(path) == expected['desk_weight.png'], 'Historical desk weight changed')
    with Image.open(path) as image:
        require(image.format == 'PNG' and image.mode == 'L' and image.size == (2048, 2048),
                'Invalid historical desk weight')
        return np.array(image)


def validate(directory):
    cfg, sources, jobs = plan()
    manifest_hash = sha(MANIFEST)
    source_hash = cfg['frozen_sources_sha256']
    check_sources(sources)
    manifest = read(MANIFEST)
    families = manifest['families']
    by_id = {j['id']: j for j in jobs}
    names = [f['id'] for f in families]
    require(len(names) == len(set(names)) == manifest['families_count'] == 480 and
            set(names) == set(by_id) and manifest['condition_pngs'] == 2880, 'Incomplete or unexpected approved cohort')
    require({p.name for p in directory.iterdir() if p.is_dir()} == set(names) | {'_provenance'},
            'Missing or unexpected family directory')
    records = []
    for index, family in enumerate(families, 1):
        job = by_id[family['id']]
        source = sources[job['source_id']]
        folder = directory / family['id']
        record = check_record(folder, family, job, source)
        audit = read(folder / 'audit.json')
        weight = historical_weight(directory, job, source, audit)
        gains = cfg['lighting'][job['light_id']]
        a, b, _, _ = check_family_pixels(folder, source, gains, weight, HISTORICAL_CONTEXTS[1:])
        current = np.array(Image.open(ROOT / 'data/inputs' / source['id'] / 'desk_weight.png')) == 0
        original = ROOT / 'data/sources' / source['source_directory'] / 'full_body'
        record['current_guard_deviation_pixels'] = sum(
            int(np.any(actual != lighting(rgb(original / f'{label}.png'), gains), axis=2)[current].sum())
            for actual, label in ((a, 'static'), (b, 'movement')))
        expected = set(family['condition_png_sha256']) | {'audit.json'}
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file() and p.name != '.DS_Store'}
        require(actual == expected, f'Unexpected files in {folder}')
        records.append(record)
        if index % 24 == 0:
            print(f'Validated {index}/480 families', flush=True)
    expected = {f"diffusion_parents/{j['base_id']}/result.json" for j in jobs} | {
        f"inputs/{s['id']}/desk_weight.png" for s in sources.values()}
    actual = {p.relative_to(directory / '_provenance').as_posix()
              for p in (directory / '_provenance').rglob('*') if p.is_file() and p.name != '.DS_Store'}
    require(actual == expected, 'Missing or unexpected historical provenance files')
    require(sha(MANIFEST) == manifest_hash and sha(ROOT / 'data/frozen_sources.json') == source_hash,
            'Manifests changed during validation; rerun against a stable snapshot')
    deviations = sum(r['current_guard_deviation_pixels'] for r in records)
    report = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), manifest_sha256=manifest_hash,
                  frozen_sources_sha256=source_hash,
                  families_checked=480, condition_pngs_checked=2880, audit_jsons_checked=480,
                  frozen_files_checked=220, historical_provenance_files_checked=len(expected),
                  historical_guard_pixel_errors=0, original_hash_mismatches=0,
                  current_guard_deviation_pixels=deviations,
                  current_guard_deviation_families=sum(r['current_guard_deviation_pixels'] > 0 for r in records),
                  current_guard_conformant=deviations == 0,
                  mask_version='Original production masks, before the coordinate correction',
                  visual_review='Historical user approval; no new human review implied',
                  known_issue_reference=17, scientific_replication_validated=False, records=records)
    write(directory / 'validation.json', report)
    return report


def hand_deviation(directory, output):
    """Archive hand pixels that differ from the frozen V7 source, within the silhouette mattes.

    The historical desk masks could reach into the hand; these pixels are what the silhouette
    condition shows. Writes one row per family and frame; does not touch validation.json.
    """
    cfg, sources, jobs = plan()
    frames = silhouette.FRAMES
    mattes = {(sid, f): silhouette.load(source, f) for sid, source in sources.items() for f in frames}
    originals = {sid: {f: rgb(ROOT / 'data/sources' / source['source_directory'] / f'full_body/{f}.png') for f in frames}
                 for sid, source in sources.items()}
    rows = []
    for job in jobs:
        residuals = []
        for frame in frames:
            alpha = mattes[(job['source_id'], frame)]
            residual = rgb(directory / job['id'] / f'full_body/{frame}.png').astype(int) - \
                lighting(originals[job['source_id']][frame], cfg['lighting'][job['light_id']])
            residuals.append(residual)
            size = np.abs(residual).max(axis=2)
            visible = alpha > 0
            rows.append(dict(family=job['id'], reference=sources[job['source_id']]['reference'], frame=frame,
                             visible_pixels=int(visible.sum()), deviating_pixels=int((size[visible] > 0).sum()),
                             deviating_opaque_pixels=int((size[alpha >= 250] > 0).sum()),
                             largest_difference=int(size[visible].max())))
        visible = (mattes[(job['source_id'], 'static')] > 0) | (mattes[(job['source_id'], 'movement')] > 0)
        rows[-1]['identical_in_both_frames'] = rows[-2]['identical_in_both_frames'] = bool(
            np.array_equal(residuals[0][visible], residuals[1][visible]))
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    output.write_text('\n'.join([','.join(columns)] + [','.join(str(r[c]) for c in columns) for r in rows]) + '\n')
    affected = sorted({r['reference'] for r in rows if r['deviating_pixels']})
    return dict(families=len(jobs), affected_families=len({r['family'] for r in rows if r['deviating_pixels']}),
                affected_references=affected, max_deviating_pixels=max(r['deviating_pixels'] for r in rows),
                max_fraction_of_visible=max(r['deviating_pixels'] / r['visible_pixels'] for r in rows),
                largest_difference=max(r['largest_difference'] for r in rows),
                identical_in_both_frames=all(r['identical_in_both_frames'] for r in rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/accepted')
    parser.add_argument('--hand-deviation', type=Path, nargs='?', const=ROOT / 'output/hand_pixel_deviation.csv',
                        help='Only count archive hand-pixel deviations under the silhouette mattes; write this CSV')
    args = parser.parse_args()
    if args.hand_deviation:
        print(hand_deviation(args.data, args.hand_deviation))
        raise SystemExit
    result = validate(args.data)
    print(f"PASS historical integrity: {result['families_checked']} families / {result['condition_pngs_checked']} PNGs / 480 audits")
    if not result['current_guard_conformant']:
        print(f"NOTE: {result['current_guard_deviation_families']} accepted families differ from the corrected current guard; preserve originals and review a fresh run separately.")
