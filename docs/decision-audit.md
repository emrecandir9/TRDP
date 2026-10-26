# Decision audit: what each decision rests on

26 October 2026, before Gate 2 and before any held-out result. Every design, analysis and modelling
decision in the [analysis spec](analysis-spec.md), [model method](model-method.md),
[baselines](baselines.md) and [controls](controls.md) is classed by what grounds it. Every reference
below was checked on 26 October 2026 against the publisher, PubMed Central or arXiv record (title,
authors, venue, year) and, where the claim depends on a finding, against the abstract.

| Class | Meaning |
| --- | --- |
| **P** | Mirrors the paper's own procedure (Matamala-Gomez et al., 2025) |
| **L** | Grounded in published evidence or theory (reference given) |
| **M** | A standard methodological principle (reference given where one exists) |
| **C** | A project choice with no literature value: declared before results, its consequences tested or to be tested by a robustness variation |
| **W** | A known weakness or limit: stated, not removable within this project |

## 1. Question, units and statistics

| Decision | Basis | Class |
| --- | --- | --- |
| Participants, not trials, are the independent units of the human target; the paper's trial-level tests are reported as inflated | Ignoring nesting inflates false positives (Aarts et al., 2014; Saravanan et al., 2020) | L |
| Primary endpoint `C_FDI`, robust on every scale; analysis spec frozen before any model result; changes only through a dated amendment log | Separating prediction from postdiction (Nosek et al., 2018) | M |
| Holm correction within secondary and exploratory families | Holm (1979) | M |
| Model instance = one initialisation × one held-out source group; 10 initialisations | A single network can mislead; inferences should rest on groups of instances (Mehrer et al., 2020) | L |
| Hierarchical bootstrap: source groups, then initialisations | Resampling at each level of nested data (Saravanan et al., 2020) | L |
| Leave one source group out; a source's 24 variants never split | Selection and test on the same data is circular (Kriegeskorte et al., 2009) | M |
| No context contrast or MEP-analogue summary on a held-out source before the evaluation; enforced in code (`evaluate.simulate_instance`) | Kriegeskorte et al. (2009) | M |
| 20 independent sources | Small samples give wide cross-validated error bars (Varoquaux, 2018); results reported per source | **W** |
| Decision rule needs every criterion with the primary encoder and P1 with both robustness encoders; the least favourable encoder is primary | Conservative by construction | C |

## 2. Stimuli

| Decision | Basis | Class |
| --- | --- | --- |
| Full body, upper limb, detached hand on black; left index abduction; trial timing | The paper's design | P |
| Detached hand as skin-only silhouette | Matches the paper's isolated hand | P |
| Synthetic (AI-generated) hands, 20 identities, one generator and scene template | Generalisation limited to hands of this kind | **W** |
| All 480 families accepted by one unblinded reviewer; rater protocol prepared, not run | Owner's decision, 24 Oct 2026 | **W** |
| Missing-wrist and arm-window controls | Wrist discontinuity reduces ownership (Tieri et al., 2015); a hand disconnected from the body weakens the virtual hand illusion (Perez-Marcos et al., 2012) | L |

## 3. Visual baselines (Gate 1)

| Decision | Basis | Class |
| --- | --- | --- |
| Frozen task-trained encoders as models of the visual input | Goal-driven networks as sensory models (Yamins & DiCarlo, 2016) | L |
| OpenCLIP ViT-bigG/14 and DINOv2 ViT-L/14 | Cherti et al. (2023); Oquab et al. (2024) | L |
| Low-level, pixel and hand-only baselines before any model | Rules out stimulus confounds first | M |

## 4. Network and training

| Decision | Basis | Class |
| --- | --- | --- |
| Leaky integrate-and-fire neurons, surrogate-gradient training in snnTorch | Neftci et al. (2019); Eshraghian et al. (2023) | L |
| τ_mem 20 ms, τ_syn 5 ms, Δt 1 ms; sensitivity 10/40 and 2/10 ms, 0.5 ms | Conventional values; sensitivity declared in the spec | L/C |
| Non-spiking twin emits the noisy LIF neuron's mean-field rate (tabulated by simulation) | Firing of noisy LIF neurons with synaptic filtering (Fourcaud & Brunel, 2002) | L |
| 256 visual, 2 × 64 motor neurons; learning rate 10⁻³; 3,000 updates | No literature value | C (robustness variations, model-method §8) |
| Visuomotor mapping learned from the observer's own connected (full-body) view only | Associative accounts of mirror responses (Heyes, 2001; Cook et al., 2014); training reshapes FDI/ADM MEP specificity (Catmur et al., 2007); computational mirror models learn such mappings (Oztop et al., 2013) | L |
| Auxiliary task: detect the index movement | The participants' counting task | P |
| Observation without execution: motor populations near rest, limits per trial (rest 1 Hz, movement 3 Hz) | Corticospinal mirror neurons are modulated during observation without muscle activity, some suppressed (Kraskov et al., 2009; Vigneswaran et al., 2013); the numeric limits are calibrations | L (principle) / C (values) |

## 5. Probe and readout (the TMS/MEP analogue)

| Decision | Basis | Class |
| --- | --- | --- |
| Resting threshold: smallest amplitude reaching a response criterion on 50% of trials, adaptive search | IFCN rule: about 50 µV in 5 of 10 trials (Rossini et al., 2015, as cited in secondary sources; the 50% rule verified) | P/L |
| Response criterion 6 spikes (≈ 10% of the population) as the analogue of 50 µV | No literature value | C |
| Task probe at 120% of threshold | The paper | P |
| Readout: population spike count 0–55 ms after the probe, minus a matched no-probe twin | MEP size reflects how many motor neurons discharge and how synchronously (Magistris et al., 1998); the human window's 5-ms start follows corticospinal conduction, which the model lacks (Di Lazzaro & Rothwell, 2014) | L (concept) / C (window) |
| ± 2 SD trimming and within-instance z-normalisation | The paper | P |
| Background-EMG exclusion → readout range rule | The paper excluded trials with background EMG | P |
| **The spiking readout falls once the population is over-driven** | In humans the MEP input–output curve is sigmoid; background contraction raises its gain 4–7× and leaves its plateau unchanged (Devanne et al., 1997). A falling readout is therefore a property of the model, not of physiology | **W** (found in this audit) |
| Spinal motoneuron stage not adopted | Recruitment all-or-none (model-method §6.15) | W |

## 6. Hypotheses and gates

| Decision | Basis | Class |
| --- | --- | --- |
| H2: gate from forearm continuity at the wrist | The paper manipulates continuity; continuity governs ownership (Tieri et al., 2015; Perez-Marcos et al., 2012); anatomically implausible hands are not self-attributed (Tsakiris & Haggard, 2005) | P/L |
| One-parameter gate, c fitted to the human DH/FB ratio on training sources | One ratio identifies one parameter | C |
| P1 and S1 close to built in for H2 | Stated per criterion (model-method §7); the informative tests are P2 (upper limb), S2 and S3(a) | **W** |
| H0 hand-only, H1 low-level salience, ungated network | Null, low-level and ablation comparisons | M |
| H2-learned and H4 | Declared comparison models (model-method §7.1) | C |
| Matched control pairs (equal area, connection differs) | Matched-pair design | M |

## 7. Health criteria and networks that fail them

| Decision | Basis | Class |
| --- | --- | --- |
| Thresholds: task AUC ≥ 0.9, ≤ 10% silent and ≤ 5% fast visual neurons, Spearman ≥ 0.9, rest 0.5–20 Hz | No literature values; declared before training | C |
| No network is excluded for a source-dependent health failure; network-level criteria (all passed by 2,400/2,400) still exclude | Excluding is never conservative: either neutral or favourable to the hypothesis (model-method §6.16; decision of 26 Oct 2026) | M |
| Over-driven held-out displays counted only if the result holds both with them removed and with them set to the readout's maximum | Worst-case (partial-identification) bounds for missing outcomes (Horowitz & Manski, 2000); the maximum mirrors the MEP plateau (Devanne et al., 1997) | L |
| Readout range valid only if the excitability curve peaks inside the tested range (sweep doubled 26 Oct 2026) | Measurement validation: the twin had not peaked in the old sweep; every premise of the rule tested on all 2,400 networks (model-method §6.17) | M |

| Each population flagged against its own readout range (ADM curve measured like the FDI one) | Applies the rule's premise to the population it concerns instead of assuming the ADM curve matches the FDI one | M |
| c fitted under each treatment; the evaluation uses the c of the same treatment | The same validity rule throughout, from fit to verdict | M |
| Learned pathways scored on training sources by cross-fitting (pathway fitted without the source) | Out-of-sample scoring, as for the held-out source; in-sample scores were not comparable (model-method §6.19) | M (fault fixed) |
| Gate 2 "different control predictions": any of 8 paired quantities, Holm-corrected, both treatments | Holm (1979); hierarchical bootstrap (Saravanan et al., 2020); the choice of quantities is the project's | C |
| Gate 2 "smooth probe responses": Spearman ≥ 0.9 over 1.0–1.5 × threshold in every network | Reuses the declared dose-response threshold | C |

Because no network is excluded on the arbitrary health thresholds of the first row any more, their
arbitrariness no longer decides which data enter the analysis.

## 8. Simulation

| Decision | Basis | Class |
| --- | --- | --- |
| Trialwise simulation for the feedforward network | Its state at the probe depends on the second frame alone (residual < e⁻¹²); identical to the continuous session without noise (test), and checked with noise on a trained network | M |

## Summary

48 decisions, counted by their first class:

| Class | Decisions |
| --- | --- |
| P, mirrors the paper | 8 |
| L, grounded in literature | 13 |
| M, methodological standard | 13 |
| C, project choice | 8 |
| W, weakness or limit | 6 |

Three of the L decisions have a grounded principle but project-chosen values (time constants, the motor
limits during observation, the readout window). Every project choice (C) was declared before the results it could affect, and none can be grounded
in a published value because none exists for this model. Their influence is tested by the declared
robustness variations (network size, learning rate, training length, time constants and step) and,
for the health thresholds, removed by the rule that no network is excluded on them.

The weaknesses (W) are reported, not removed: 20 synthetic sources, one unblinded reviewer, P1 and S1
close to built in for H2, an abstract readout whose over-driven behaviour is not physiological, and no
valid spinal stage. One was found by this audit: the falling over-driven readout contradicts the human
input–output curve. The rule for over-driven displays handles it conservatively, and it must be stated
as a model limitation.

## References

- Aarts, E., Verhage, M., Veenvliet, J. V., Dolan, C. V., & van der Sluis, S. (2014). A solution to dependency: using multilevel analysis to accommodate nested data. *Nature Neuroscience*, 17(4), 491–496. doi:10.1038/nn.3648
- Catmur, C., Walsh, V., & Heyes, C. (2007). Sensorimotor learning configures the human mirror system. *Current Biology*, 17(17), 1527–1531.
- Cherti, M., et al. (2023). Reproducible scaling laws for contrastive language-image learning. *CVPR*, 2818–2829. doi:10.1109/CVPR52729.2023.00276
- Cook, R., Bird, G., Catmur, C., Press, C., & Heyes, C. (2014). Mirror neurons: from origin to function. *Behavioral and Brain Sciences*, 37(2), 177–192. doi:10.1017/S0140525X13000903
- Devanne, H., Lavoie, B. A., & Capaday, C. (1997). Input-output properties and gain changes in the human corticospinal pathway. *Experimental Brain Research*, 114(2), 329–338. doi:10.1007/pl00005641
- Di Lazzaro, V., & Rothwell, J. C. (2014). Corticospinal activity evoked and modulated by non-invasive stimulation of the intact human motor cortex. *Journal of Physiology*, 592(19), 4115–4128.
- Eshraghian, J. K., et al. (2023). Training spiking neural networks using lessons from deep learning. *Proceedings of the IEEE*, 111(9), 1016–1054. doi:10.1109/JPROC.2023.3308088
- Fadiga, L., Fogassi, L., Pavesi, G., & Rizzolatti, G. (1995). Motor facilitation during action observation: a magnetic stimulation study. *Journal of Neurophysiology*, 73, 2608–2611.
- Fourcaud, N., & Brunel, N. (2002). Dynamics of the firing probability of noisy integrate-and-fire neurons. *Neural Computation*, 14(9), 2057–2110. doi:10.1162/089976602320264015
- Heyes, C. (2001). Causes and consequences of imitation. *Trends in Cognitive Sciences*, 5(6), 253–261. doi:10.1016/S1364-6613(00)01661-2
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. JSTOR 4615733
- Horowitz, J. L., & Manski, C. F. (2000). Nonparametric analysis of randomized experiments with missing covariate and outcome data. *Journal of the American Statistical Association*, 95, 77–84.
- Kraskov, A., Dancause, N., Quallo, M. M., Shepherd, S., & Lemon, R. N. (2009). Corticospinal neurons in macaque ventral premotor cortex with mirror properties: a potential mechanism for action suppression? *Neuron*, 64(6), 922–930. doi:10.1016/j.neuron.2009.12.010
- Kriegeskorte, N., Simmons, W. K., Bellgowan, P. S. F., & Baker, C. I. (2009). Circular analysis in systems neuroscience: the dangers of double dipping. *Nature Neuroscience*, 12(5), 535–540. doi:10.1038/nn.2303
- Magistris, M. R., Rösler, K. M., Truffert, A., & Myers, J. P. (1998). Transcranial stimulation excites virtually all motor neurons supplying the target muscle. *Brain*, 121(3), 437–450.
- Matamala-Gomez, M., Frisco, F., et al. (2025). Virtual body continuity during action observation affects motor cortical excitability. *Scientific Reports*, 15, 13364. doi:10.1038/s41598-025-97695-9
- Mehrer, J., Spoerer, C. J., Kriegeskorte, N., & Kietzmann, T. C. (2020). Individual differences among deep neural network models. *Nature Communications*, 11, 5725. doi:10.1038/s41467-020-19632-w
- Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. *IEEE Signal Processing Magazine*, 36(6), 51–63. doi:10.1109/MSP.2019.2931595
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *PNAS*, 115(11), 2600–2606. doi:10.1073/pnas.1708274114
- Oquab, M., Darcet, T., et al. (2024). DINOv2: learning robust visual features without supervision. *Transactions on Machine Learning Research*. arXiv:2304.07193
- Oztop, E., Kawato, M., & Arbib, M. A. (2013). Mirror neurons: functions, mechanisms and models. *Neuroscience Letters*, 540, 43–55.
- Perez-Marcos, D., Sanchez-Vives, M. V., & Slater, M. (2012). Is my hand connected to my body? The impact of body continuity and arm alignment on the virtual hand illusion. *Cognitive Neurodynamics*, 6(4), 295–305. doi:10.1007/s11571-011-9178-5
- Rossini, P. M., et al. (2015). Non-invasive electrical and magnetic stimulation of the brain, spinal cord, roots and peripheral nerves: basic principles and procedures for routine clinical and research application. An updated report from an I.F.C.N. Committee. *Clinical Neurophysiology*, 126(6), 1071–1107. doi:10.1016/j.clinph.2015.02.001
- Saravanan, V., Berman, G. J., & Sober, S. J. (2020). Application of the hierarchical bootstrap to multi-level data in neuroscience. *Neurons, Behavior, Data Analysis, and Theory*, 3(5).
- Tieri, G., Tidoni, E., Pavone, E. F., & Aglioti, S. M. (2015). Body visual discontinuity affects feeling of ownership and skin conductance responses. *Scientific Reports*, 5, 17139. doi:10.1038/srep17139
- Tsakiris, M., & Haggard, P. (2005). The rubber hand illusion revisited: visuotactile integration and self-attribution. *Journal of Experimental Psychology: Human Perception and Performance*, 31(1), 80–91. doi:10.1037/0096-1523.31.1.80
- Varoquaux, G. (2018). Cross-validation failure: small sample sizes lead to large error bars. *NeuroImage*, 180, 68–77.
- Vigneswaran, G., Philipp, R., Lemon, R. N., & Kraskov, A. (2013). M1 corticospinal mirror neurons and their role in movement suppression during action observation. *Current Biology*, 23(3), 236–243. doi:10.1016/j.cub.2012.12.006
- Yamins, D. L. K., & DiCarlo, J. J. (2016). Using goal-driven deep learning models to understand sensory cortex. *Nature Neuroscience*, 19(3), 356–365.
