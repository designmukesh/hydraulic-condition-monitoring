"""Build PROJECT_GUIDE.html — self-contained, all figures embedded as base64."""
import os, base64

ROOT = "/home/hatch/workspace/portfolio/01-hydraulic-condition-monitoring"
FIG = os.path.join(ROOT, "reports/figures")

def img(name, caption):
    with open(os.path.join(FIG, name), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')

CSS = """
:root { --accent:#1b6b4a; --accent-light:#e8f3ee; --ink:#1e2a24; --muted:#5c6f65;
        --line:#dce7e1; --code-bg:#f4f7f5; --callout:#fdf6e3; }
* { box-sizing:border-box; }
body { font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
       color:var(--ink); line-height:1.7; margin:0; background:#fff; }
.wrap { max-width:860px; margin:0 auto; padding:0 24px 80px; }
header.hero { background:linear-gradient(135deg,#123f2c,#1b6b4a); color:#fff;
              padding:56px 24px 44px; }
header.hero .wrap { padding-bottom:0; }
header.hero h1 { margin:0 0 12px; font-size:2rem; line-height:1.25; }
header.hero p.sub { margin:0; opacity:.88; font-size:1.05rem; max-width:640px; }
.badges { margin-top:18px; display:flex; flex-wrap:wrap; gap:8px; }
.badges span { background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.35);
               border-radius:20px; padding:4px 14px; font-size:.82rem; }
nav.toc { background:var(--accent-light); border:1px solid var(--line);
          border-radius:10px; padding:18px 24px; margin:28px 0; }
nav.toc h3 { margin:0 0 8px; font-size:.95rem; color:var(--accent); }
nav.toc ol { margin:0; padding-left:20px; columns:2; font-size:.9rem; }
nav.toc a { color:var(--ink); text-decoration:none; }
nav.toc a:hover { color:var(--accent); text-decoration:underline; }
h2 { color:var(--accent); border-bottom:2px solid var(--line);
     padding-bottom:8px; margin-top:52px; font-size:1.45rem; }
h3 { margin-top:30px; font-size:1.1rem; }
figure { margin:26px 0; text-align:center; }
figure img { max-width:100%; border:1px solid var(--line); border-radius:8px; }
figcaption { font-size:.85rem; color:var(--muted); margin-top:8px; }
table { border-collapse:collapse; width:100%; margin:18px 0; font-size:.92rem; }
th, td { border:1px solid var(--line); padding:9px 12px; text-align:left; }
th { background:var(--accent-light); color:var(--accent); }
tr:nth-child(even) td { background:#fafcfb; }
pre { background:var(--code-bg); border:1px solid var(--line); border-radius:8px;
      padding:16px 18px; overflow-x:auto; font-size:.85rem; line-height:1.55; }
code { font-family:'SF Mono',Menlo,Consolas,monospace; }
p code, li code, td code { background:var(--code-bg); padding:2px 6px;
      border-radius:4px; font-size:.86em; }
.callout { background:var(--callout); border-left:4px solid #d9a521;
           padding:14px 18px; border-radius:0 8px 8px 0; margin:22px 0; font-size:.95rem; }
.callout.product { background:#eef4ff; border-left-color:#2b5fc7; }
.callout strong.tag { display:block; font-size:.78rem; letter-spacing:.06em;
           text-transform:uppercase; margin-bottom:4px; color:var(--muted); }
.qa { margin:16px 0; }
.qa .q { font-weight:700; }
.qa .a { margin:4px 0 0 0; }
footer { margin-top:60px; padding-top:20px; border-top:1px solid var(--line);
         color:var(--muted); font-size:.85rem; }
@media print { header.hero { -webkit-print-color-adjust:exact; print-color-adjust:exact; } }
"""

def section(num, title, body):
    return f'<h2 id="s{num}">{num}. {title}</h2>\n{body}\n'

TOC = ["The problem — start here, not with code", "The data", "EDA: do the signals separate the faults?",
       "Feature engineering: 43,680 numbers → 187 features", "Modeling: four small problems, not one big one",
       "Reading the results like a maintenance planner", "Which sensors matter: the product insight",
       "Limitations — stated before anyone asks", "From model to product",
       "Interview prep: questions I'd expect", "Sharing kit: LinkedIn + Medium"]

toc_html = '<nav class="toc"><h3>Contents</h3><ol>' + "".join(
    f'<li><a href="#s{i+1}">{t}</a></li>' for i, t in enumerate(TOC)) + '</ol></nav>'

S1 = """
<p>Unplanned hydraulic failures stop production lines. A failed cooler or a leaking pump doesn't just cost a spare part — it costs the unplanned shutdown, the rush technician visit, and the production that didn't happen.</p>
<p>But most ML tutorials miss the point: <strong>"something is wrong" is not an actionable prediction.</strong> A maintenance planner needs three answers:</p>
<ol>
<li><strong>Which component</strong> is degrading — cooler, valve, pump, or accumulator?</li>
<li><strong>How badly</strong> — full efficiency, reduced, or close to failure?</li>
<li><strong>What should I do about it</strong> — and when?</li>
</ol>
<p>This project answers the first two. (The third becomes Project #3 in my portfolio: a maintenance <em>decision</em> engine.)</p>
<div class="callout product"><strong class="tag">Product lens</strong>Anyone can train a classifier. A product person starts from the <em>decision the prediction enables</em>. That framing is what this whole project is built on.</div>
""" + img("05_workflow.png", "Project workflow — every stage answers a question before the next begins.")

S2 = """
<p>The dataset comes from a hydraulic test rig at a research lab, published by Helwig et al. to the <strong>UCI Machine Learning Repository</strong> (2018). It holds <strong>2,205 operating cycles</strong>, each 60 seconds long, recorded under systematically varied component health states. Because fault states were <em>induced deliberately</em>, the labels are trustworthy — the closest public analog to labeled industrial asset data, which is notoriously hard to get.</p>
<div class="callout"><strong class="tag">Independence note</strong>Public research data; not affiliated with any employer; no proprietary data used. I state this in every portfolio project.</div>
<h3>Sensor inventory — 17 sensors, 3 sampling rates</h3>
<table><tr><th>Sensors</th><th>Measures</th><th>Rate</th><th>Samples/cycle</th></tr>
<tr><td>PS1–PS6</td><td>Pressure (bar)</td><td>100 Hz</td><td>6,000 each</td></tr>
<tr><td>EPS1</td><td>Motor power (W)</td><td>100 Hz</td><td>6,000</td></tr>
<tr><td>FS1–FS2</td><td>Volume flow (l/min)</td><td>10 Hz</td><td>600 each</td></tr>
<tr><td>TS1–TS4</td><td>Temperature (°C)</td><td>1 Hz</td><td>60 each</td></tr>
<tr><td>VS1</td><td>Vibration (mm/s)</td><td>1 Hz</td><td>60</td></tr>
<tr><td>SE / CE / CP</td><td>Efficiency / cooling efficiency / cooling power</td><td>1 Hz</td><td>60 each</td></tr></table>
<p>Total: <strong>43,680 raw values per cycle</strong> — which immediately tells you no classifier ingests this raw. Compression comes first (Section 4).</p>
<h3>Targets — 4 components, each with its own health states</h3>
<table><tr><th>Component</th><th>States</th><th>Cycles</th></tr>
<tr><td>Cooler</td><td>3 close to failure / 20 reduced efficiency / 100 full efficiency</td><td>732 / 732 / 741</td></tr>
<tr><td>Valve</td><td>73 close to failure / 80 severe lag / 90 small lag / 100 optimal</td><td>360 / 360 / 360 / 1125</td></tr>
<tr><td>Pump</td><td>0 no leakage / 1 weak leakage / 2 severe leakage</td><td>1221 / 492 / 492</td></tr>
<tr><td>Accumulator</td><td>90 close to failure / 100 severely reduced / 115 slightly reduced / 130 optimal</td><td>808 / 399 / 399 / 599</td></tr></table>
<p>Two pre-modeling observations: (1) cooler is near-balanced but the rest skew healthy — so I evaluate with <strong>macro-F1</strong>, forcing every fault state to count equally; (2) every class has ≥360 cycles, so stratified splits work.</p>
<h3>Setup — exact commands</h3>
<pre><code>cd 01-hydraulic-condition-monitoring
python3 -m venv .venv &amp;&amp; source .venv/bin/activate
pip install -r requirements.txt
python data/download_data.py      # fetches the public UCI archive (~500 MB)
jupyter notebook notebooks/</code></pre>
"""

S3 = """
<p><strong>The single most important EDA question:</strong> if I plot the average sensor trace per fault class, do the curves separate? If yes, a classifier has something real to learn. If no, stop.</p>
<h3>Class distributions</h3>
""" + img("01_target_distributions.png", "Component health states across 2,205 test-rig cycles.") + """
<h3>The physics check</h3>
<p>I tested two physical hypotheses against the raw signals:</p>
<ul>
<li><strong>Hypothesis A:</strong> a degraded cooler can't reject heat → oil temperature sensors (TS3/TS4) should run hotter as cooler condition worsens.</li>
<li><strong>Hypothesis B:</strong> a lagging valve changes switching dynamics → pressure sensors (PS1/PS2) should show reshaped plateaus.</li>
</ul>
""" + img("02_signal_separation.png", "Mean sensor traces split by component health — classes visibly separate.") + """
<div class="callout product"><strong class="tag">What this chart is for</strong><strong>Hypothesis A confirmed, dramatically:</strong> a cooler "close to failure" runs at ~56°C vs ~39°C at full efficiency — a <strong>~17°C separation</strong> visible to the naked eye. <strong>Hypothesis B confirmed:</strong> valve lag visibly reshapes pressure plateaus. Before showing any model score, I can show a plant manager this image and say: <em>"Your sensors already see the failure coming. We're just teaching a computer to watch them 24/7."</em> That sentence is worth more than any F1 score in a stakeholder meeting.</div>
"""

S4 = """
<p>43,680 values per cycle can't go into a classifier directly. The standard, honest baseline for this benchmark: compress each cycle into <strong>statistical descriptors</strong> — transparent, explainable, and fast.</p>
""" + img("06_feature_anatomy.png", "One real TS3 oil-temperature cycle per cooler state, with the statistics extracted from it.") + """
<h3>The 11 features per sensor (each explainable in one sentence)</h3>
<table><tr><th>Feature</th><th>Physical meaning</th></tr>
<tr><td><code>mean</code></td><td>Operating level (e.g. steady oil temperature)</td></tr>
<tr><td><code>std</code></td><td>Fluctuation around the level</td></tr>
<tr><td><code>min</code> / <code>max</code> / <code>median</code></td><td>Extremes; median is the spike-robust level</td></tr>
<tr><td><code>ptp</code></td><td>Peak-to-peak swing</td></tr>
<tr><td><code>rms</code> / <code>energy</code></td><td>Signal energy level</td></tr>
<tr><td><code>crest</code></td><td>Peak ÷ RMS — spikiness: transients and impacts</td></tr>
<tr><td><code>skew</code></td><td>Asymmetry — e.g. one-sided pressure spikes</td></tr>
<tr><td><code>kurt</code></td><td>Tailedness — impulsive events</td></tr></table>
<p>17 sensors × 11 features = <strong>187 features per cycle</strong>. The implementation (<code>src/data.py:build_feature_frame()</code>) processes sensors one file at a time — the raw files total ~500 MB, and the boring engineering detail of not blowing memory is exactly what interviewers like to hear.</p>
<pre><code>def signal_features(x):
    x = np.asarray(x, dtype=float)
    rms = np.sqrt(np.mean(x ** 2))
    return {"mean": np.mean(x), "std": np.std(x),
            "min": np.min(x), "max": np.max(x),
            "median": np.median(x), "ptp": np.ptp(x),
            "rms": rms, "crest": np.max(np.abs(x)) / (rms + 1e-12),
            "skew": stats.skew(x), "kurt": stats.kurtosis(x),
            "energy": np.mean(x ** 2)}</code></pre>
"""

S5 = """
<h3>Decision 1 — one classifier per component</h3>
<p>The rig varies components <strong>independently</strong>, so failure modes are independent. Four focused models are simpler to debug and map 1:1 to work orders ("the <em>cooler</em> needs attention"). When the problem decomposes naturally, decompose the model too.</p>
<h3>Decision 2 — RandomForest vs HistGradientBoosting, winner per component</h3>
<p>Tree ensembles: no normalization needed, robust to correlated features, and they give feature importances for free.</p>
<pre><code>for t in TARGETS:  # cooler, valve, pump, accumulator
    X_train, X_test, y_train, y_test = train_test_split(
        X, y[t].values, test_size=0.2, random_state=42,
        stratify=y[t].values)          # every fault state in both sets
    for name, model in [("rf", RandomForestClassifier(
                                n_estimators=300, n_jobs=-1, random_state=42)),
                        ("hgb", HistGradientBoostingClassifier(random_state=42))]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        # record accuracy AND macro-F1 for every combination</code></pre>
<h3>Decision 3 — no leakage, by construction</h3>
<div class="callout"><strong class="tag">Discipline that matters</strong>Features are computed <strong>per cycle</strong> and the split is <strong>by cycle</strong> (stratified 80/20, seed 42). No cycle appears in both sets. Leakage through overlapping time windows is the #1 silent killer of time-series ML credibility — stating the discipline explicitly is what separates a careful practitioner from a notebook hobbyist.</div>
"""

S6 = """
<table><tr><th>Component</th><th>States</th><th>Best model</th><th>Accuracy</th><th>Macro-F1</th></tr>
<tr><td>Cooler</td><td>3</td><td>RF / HGB</td><td><strong>1.000</strong></td><td><strong>1.000</strong></td></tr>
<tr><td>Pump leakage</td><td>3</td><td>RandomForest</td><td><strong>0.998</strong></td><td><strong>0.997</strong></td></tr>
<tr><td>Valve</td><td>4</td><td>RandomForest</td><td><strong>0.986</strong></td><td><strong>0.983</strong></td></tr>
<tr><td>Accumulator</td><td>4</td><td>HistGradientBoosting</td><td><strong>0.986</strong></td><td><strong>0.985</strong></td></tr></table>
<div class="callout"><strong class="tag">Honest read</strong>These are strong — almost suspiciously strong. The truth: <strong>test-rig data with deliberately induced, well-separated faults is the easy case.</strong> Real plants add drift, dropouts, and mixed modes. I volunteer that before being asked — it builds more credibility than perfect scores do.</div>
""" + img("03_confusion_matrices.png", "Confusion matrices on held-out cycles (20% stratified split).") + """
<h3>Reading these as a maintenance planner</h3>
<ul>
<li><strong>Cooler:</strong> clean diagonal — a "reduced efficiency" flag is trustworthy enough to <em>schedule</em> service, not just investigate.</li>
<li><strong>Valve / accumulator:</strong> rare confusions sit between <em>adjacent</em> severity states — the good kind of error. The action (inspect the valve) is the same either way.</li>
<li><strong>The expensive error</strong> — predicting "healthy" when truth is "close to failure" — is essentially empty across all four matrices. That's the cell that prevents unplanned shutdowns, and it's what I'd monitor in production.</li>
</ul>
"""

S7 = """
<p>Feature importance aggregated to the <strong>sensor</strong> level (from the random forest — the interpretable model):</p>
""" + img("04_sensor_importance.png", "Sensor-level importance: physics shows up where it should.") + """
<table><tr><th>Component</th><th>Top sensors</th><th>Physical sense</th></tr>
<tr><td>Cooler</td><td>CE, CP, TS4</td><td>Cooling efficiency/power + temperature — the heat-rejection loop</td></tr>
<tr><td>Valve</td><td>PS2, SE, FS1</td><td>Pressure + flow — the switching dynamics</td></tr>
<tr><td>Pump</td><td>SE, FS1, EPS1</td><td>Efficiency + flow + motor power — a leaking pump moves less fluid per watt</td></tr>
<tr><td>Accumulator</td><td>SE, FS2, FS1, PS3</td><td>Efficiency + flow + pressure — the storage/discharge loop</td></tr></table>
<div class="callout product"><strong class="tag">The product insight</strong>This chart answers a <strong>budget</strong> question, not a modeling question. If temperature sensors dominate cooler prediction, a plant can retrofit <em>just those</em> and skip the expensive 100 Hz pressure array. <strong>ML tells you what to predict; feature importance tells you what to buy.</strong> That's the sentence I'd use with a VP of Operations.</div>
"""

S8 = """
<ol>
<li><strong>Test rig ≠ plant floor.</strong> Next: robustness tests with synthetic sensor noise and missing-data regimes.</li>
<li><strong>Single split, single seed.</strong> Next: repeated stratified CV with confidence intervals.</li>
<li><strong>Fixed 60-second framing.</strong> A streaming deployment needs windowing, latency budgets, and alert debouncing — otherwise you page someone 60 times for one degrading cooler.</li>
<li><strong>No spectral features yet.</strong> The 100 Hz pressure/power signals deserve FFT-band features.</li>
<li><strong>No sequence-model challenger.</strong> A 1D-CNN/LSTM on raw signals is the natural next comparison.</li>
</ol>
"""

S9 = """
<p>A model in a notebook saves nobody money. Here's the productization path — and how I'd discuss it in a Director/Head/AI-Lead interview:</p>
""" + img("07_decision_flow.png", "From prediction to product: the maintenance decision flow.") + """
<ul>
<li><strong>Users &amp; decisions:</strong> reliability engineers and maintenance planners. Output must land in the existing CMMS/work-order flow — not another dashboard nobody opens.</li>
<li><strong>KPIs that matter:</strong> not F1 — <em>unplanned downtime hours avoided</em>, <em>false-alarm rate per 100 cycles</em> (alert fatigue kills adoption), <em>mean lead time</em> (cycles of warning before "close to failure").</li>
<li><strong>Rollout:</strong> shadow mode beside human planners for 4–6 weeks → "recommend" → auto-raise low-risk work orders. Trust is deployed in stages.</li>
<li><strong>The cost layer (→ Project #3):</strong> <em>P(failure) × cost of unplanned downtime</em> vs <em>cost of planned intervention</em> turns "pump: weak leakage" into <strong>repair now / inspect later / continue</strong>. That's where the MBA meets the ML.</li>
</ul>
"""

S10 = """
<div class="qa"><p class="q">Q: Why one model per component instead of multi-output?</p>
<p class="a">A: Components degrade independently, so failure modes are independent. Separate models debug cleanly, map 1:1 to work orders, and let me pick the best algorithm per component.</p></div>
<div class="qa"><p class="q">Q: Why macro-F1 instead of accuracy?</p>
<p class="a">A: Accuracy lets a model hide behind the majority healthy class (valve is 51% "optimal"). Macro-F1 forces every fault state to count equally — missing a rare "close to failure" is the expensive error.</p></div>
<div class="qa"><p class="q">Q: How did you avoid leakage?</p>
<p class="a">A: Features per cycle, split by cycle, stratified, fixed seed. With time-series I'd also watch for overlapping windows leaking across the split; here cycles are independent by construction.</p></div>
<div class="qa"><p class="q">Q: Why statistical features instead of deep learning?</p>
<p class="a">A: It's the honest baseline — explainable to the engineers who must trust it, trains in seconds, and scores are near-ceiling. I'd still run a 1D-CNN challenger next; I list that explicitly rather than pretending the baseline is final.</p></div>
<div class="qa"><p class="q">Q: Cooler accuracy is 1.0 — overfitting?</p>
<p class="a">A: Easy case, not overfitting: test-rig data, induced faults, 17°C naked-eye separation. I'd say this upfront — the real test is plant data with drift.</p></div>
<div class="qa"><p class="q">Q: How would you deploy it?</p>
<p class="a">A: Shadow mode → recommend → auto-raise low-risk work orders. Instrument false-alarm rate and lead time. Feed predictions into a cost-based decision engine (my next project).</p></div>
<div class="qa"><p class="q">Q: What with twice the time?</p>
<p class="a">A: Spectral features, sequence-model challenger, repeated CV, missing-data robustness, streaming/windowing design.</p></div>
"""

S11 = """
<h3>LinkedIn post (adapt freely)</h3>
<pre><code>Unplanned hydraulic failures don't start as failures —
they start as a cooler running 17°C hotter than it should.

I built a condition-monitoring project on public test-rig data
(2,205 cycles, 17 sensors) that predicts WHICH component is
degrading — cooler, valve, pump, accumulator — and how badly.

The most interesting finding wasn't the 98%+ scores. It was that
the model rediscovers the physics: temperature sensors dominate
cooler prediction, pressure sensors dominate valve prediction.
ML tells you what to predict; feature importance tells you where
to spend instrumentation budget.

Full write-up + code: [repo link]
Next: turning predictions into maintenance DECISIONS with a
cost-based decision engine.

#PredictiveMaintenance #IndustrialAI #MachineLearning</code></pre>
<h3>Medium article outline</h3>
<ol>
<li>The maintenance planner's real question</li>
<li>What the data looks like — and the 17°C chart</li>
<li>From 43,680 numbers to 187 features</li>
<li>Four small models, honest evaluation</li>
<li>The sensor-importance insight</li>
<li>Limitations I'd state before you ask</li>
<li>From notebook to work order: productizing it</li>
</ol>
"""

sections = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11]
body = toc_html + "".join(section(i + 1, t, s)
                          for i, (t, s) in enumerate(zip(TOC, sections)))

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hydraulic System Condition Monitoring — Complete Project Guide</title>
<style>{CSS}</style></head>
<body>
<header class="hero"><div class="wrap">
<h1>Hydraulic System Condition Monitoring</h1>
<p class="sub">The complete project guide — how I built it, why each step exists, and what every result means. Written to be read, learned, defended in interviews, and adapted into articles.</p>
<div class="badges"><span>2,205 cycles</span><span>17 sensors</span><span>187 features</span>
<span>4 classifiers</span><span>98–100% accuracy</span><span>Public UCI data</span></div>
</div></header>
<div class="wrap">
{body}
<footer>Built with Python, scikit-learn, pandas, matplotlib. Data: Helwig et al., UCI Machine Learning Repository (2018) — public research data; independent portfolio project, no proprietary data used.</footer>
</div></body></html>"""

out = os.path.join(ROOT, "PROJECT_GUIDE.html")
with open(out, "w") as f:
    f.write(html)
print("wrote", out, f"({os.path.getsize(out)/1024:.0f} KB)")
