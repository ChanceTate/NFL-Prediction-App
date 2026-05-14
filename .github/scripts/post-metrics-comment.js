// Posts (or updates) a PR comment with model metrics, per-fold breakdown,
// and feature analysis. Detects MAE regressions vs the baseline run on main
// and fails the job unless the PR carries the `skip-metrics-check` label.
//
// MAIN_RUN_ID env var = the baseline run id from find-main-run, '' if none.

const fs = require("fs");

const MARKER = "<!-- metrics-comment -->";
const TRACKED_FOR_REGRESSION = ["LinearRegression", "LightGBM"];
const SKIP_LABEL = "skip-metrics-check";

const loadJson = (path) => JSON.parse(fs.readFileSync(path, "utf8"));

const tryLoadBaseline = (core) => {
  try {
    const raw = loadJson("baseline/metrics.json");
    // Only the walk-forward structure has an "aggregate" key. Older
    // single-fold baselines are skipped so we don't compare apples to oranges.
    if (raw && raw.aggregate) return raw;
    core.info("Baseline metrics.json predates walk-forward CV; skipping comparison.");
  } catch (e) {
    core.info("No baseline metrics.json available; first comparison.");
  }
  return null;
};

// Delta is "noise" if its magnitude is below half of main's stddev. Outside
// the band, label as improvement or regression with an emoji.
const fmtDelta = (delta, baselineStd, lowerIsBetter) => {
  if (delta === null || delta === undefined) return "—";
  const sign = delta > 0 ? "+" : "";
  const noiseBand = 0.5 * (baselineStd || 0);
  const isNoise = Math.abs(delta) <= noiseBand;
  const isRegression = lowerIsBetter ? delta > 0 : delta < 0;
  const emoji = isNoise ? "" : isRegression ? " ⚠️" : " ✅";
  return `${sign}${delta.toFixed(2)}${emoji}`;
};

const buildAggregateTable = (currentAggregate, findBaseline) => {
  const rows = currentAggregate.map((m) => {
    const b = findBaseline(m.label);
    const maeDelta = b ? m.mae_mean - b.mae_mean : null;
    const r2Delta = b ? m.r2_mean - b.r2_mean : null;
    return (
      `| ${m.label} | ${m.mae_mean.toFixed(2)} ± ${m.mae_std.toFixed(2)} | ` +
      `${fmtDelta(maeDelta, b ? b.mae_std : 0, true)} | ${m.mae_max.toFixed(2)} | ` +
      `${m.r2_mean.toFixed(2)} ± ${m.r2_std.toFixed(2)} | ` +
      `${fmtDelta(r2Delta, b ? b.r2_std : 0, false)} |`
    );
  });
  return (
    `| Model | MAE | Δ MAE | Worst fold | R² | Δ R² |\n` +
    `|---|---|---|---|---|---|\n` +
    rows.join("\n")
  );
};

const buildPerFoldTable = (currentFolds) => {
  const seasons = currentFolds.map((f) => f.test_season);
  const modelLabels = TRACKED_FOR_REGRESSION.concat(["Baseline (mean)"]);
  const header = `| Model | ${seasons.join(" | ")} |`;
  const sep = `|---${"|---".repeat(seasons.length)}|`;
  const rows = modelLabels.map((label) => {
    const cells = currentFolds.map((f) => {
      const m = f.models.find((x) => x.label === label);
      return m ? m.mae.toFixed(2) : "—";
    });
    return `| ${label} | ${cells.join(" | ")} |`;
  });
  return `${header}\n${sep}\n${rows.join("\n")}`;
};

// One table per model with permutation, ablation Δ, and per-fold sign count.
const buildAnalysisBlock = (core) => {
  let importance;
  try {
    importance = loadJson("importance.json");
  } catch (e) {
    core.info("No importance.json available; skipping feature analysis.");
    return "";
  }
  if (importance.length === 0) return "";

  let ablation = [];
  try {
    ablation = loadJson("ablation.json");
  } catch (e) {
    core.info("No ablation.json available; rendering permutation only.");
  }

  const sections = importance.map((impEntry) => {
    const label = impEntry.label;
    const impLookup = Object.fromEntries(
      impEntry.importance.map((i) => [i.feature, i.value])
    );
    const ablEntry = ablation.find((a) => a.label === label);
    const ablRows = ablEntry ? ablEntry.ablation : [];
    const nFolds = ablEntry ? ablEntry.n_folds : null;

    // If ablation is present, order rows by ablation Δ descending. Without it,
    // fall back to permutation order.
    const rows = ablEntry
      ? ablRows.map((row) => {
          const perm = impLookup[row.feature] ?? 0;
          const sign = row.delta_mean >= 0 ? "+" : "";
          const folds = nFolds ? `${row.n_positive}/${nFolds}` : "—";
          return (
            `| ${row.feature} | ${perm.toFixed(2)} | ` +
            `${sign}${row.delta_mean.toFixed(2)} ± ${row.delta_std.toFixed(2)} | ${folds} |`
          );
        })
      : impEntry.importance.map(
          (i) => `| ${i.feature} | ${i.value.toFixed(2)} | — | — |`
        );

    return (
      `### ${label}\n\n` +
      `| Feature | Permutation | Ablation Δ MAE | Folds + |\n` +
      `|---|---|---|---|\n` +
      rows.join("\n")
    );
  });

  return (
    `\n\n## Feature analysis (mean across folds)\n\n` +
    `Permutation = how much the trained model uses each feature ` +
    `(MAE worsening when its values are shuffled). Ablation Δ = ` +
    `how much MAE worsens if the model is retrained without it. ` +
    `Folds + = how many folds had a positive ablation Δ (sign ` +
    `consistency, more reliable than mean alone when std is large). ` +
    `Disagreement between permutation and ablation is informative: ` +
    `high permutation + low ablation = redundant; low permutation + ` +
    `high ablation = unique contributor.\n\n` +
    sections.join("\n\n")
  );
};

// Fail only when mean MAE worsens beyond half of main's fold-to-fold stddev.
// Within that band, the change is statistically indistinguishable from
// "we got a different luck-of-the-draw on the test seasons."
const detectRegressions = (currentAggregate, findBaseline) => {
  const out = [];
  for (const m of currentAggregate) {
    if (!TRACKED_FOR_REGRESSION.includes(m.label)) continue;
    const b = findBaseline(m.label);
    if (!b) continue;
    const delta = m.mae_mean - b.mae_mean;
    const noiseBand = 0.5 * (b.mae_std || 0);
    if (delta > noiseBand) {
      out.push(
        `${m.label} mean MAE worsened by ${delta.toFixed(2)} ` +
          `(${b.mae_mean.toFixed(2)} → ${m.mae_mean.toFixed(2)}), ` +
          `beyond noise band of ±${noiseBand.toFixed(2)} (0.5× main std).`
      );
    }
  }
  return out;
};

const upsertComment = async (github, context, body) => {
  const { data: comments } = await github.rest.issues.listComments({
    owner: context.repo.owner,
    repo: context.repo.repo,
    issue_number: context.issue.number,
  });
  const existing = comments.find((c) => c.body && c.body.includes(MARKER));

  if (existing) {
    await github.rest.issues.updateComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      comment_id: existing.id,
      body,
    });
  } else {
    await github.rest.issues.createComment({
      owner: context.repo.owner,
      repo: context.repo.repo,
      issue_number: context.issue.number,
      body,
    });
  }
};

module.exports = async ({ github, context, core }) => {
  const current = loadJson("metrics.json");
  const currentAggregate = current.aggregate || [];
  const currentFolds = current.folds || [];

  const baseline = tryLoadBaseline(core);
  const baselineAggregate = baseline ? baseline.aggregate : null;
  const findBaseline = (label) =>
    baselineAggregate ? baselineAggregate.find((b) => b.label === label) : null;

  const mainRunId = process.env.MAIN_RUN_ID || "";
  const baselineNote = baselineAggregate
    ? `Compared against main run [#${mainRunId}](${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${mainRunId}).`
    : "_No baseline available; this is likely the first run, or main predates walk-forward CV._";

  const regressions = detectRegressions(currentAggregate, findBaseline);
  const prLabels = (context.payload.pull_request?.labels || []).map((l) => l.name);
  const skip = prLabels.includes(SKIP_LABEL);

  let regressionBlock = "";
  if (regressions.length > 0) {
    regressionBlock =
      `\n\n### ⚠️ Regression detected\n\n` +
      regressions.map((r) => `- ${r}`).join("\n") +
      (skip
        ? `\n\n_Skipped by \`${SKIP_LABEL}\` label._`
        : `\n\nAdd the \`${SKIP_LABEL}\` label if this is intentional.`);
  }

  const body =
    `${MARKER}\n## Model metrics (walk-forward CV)\n\n` +
    `### Aggregate (mean ± std across ${currentFolds.length} folds)\n\n` +
    buildAggregateTable(currentAggregate, findBaseline) +
    `\n\nΔ is flagged ✅/⚠️ only when its magnitude exceeds 0.5× main's fold-to-fold std (the noise band).\n\n` +
    `### Per-fold MAE\n\n` +
    buildPerFoldTable(currentFolds) +
    `\n\n` +
    baselineNote +
    regressionBlock +
    buildAnalysisBlock(core);

  await upsertComment(github, context, body);

  if (regressions.length > 0 && !skip) {
    core.setFailed(
      `MAE regression detected. Add '${SKIP_LABEL}' label to override.\n\n${regressions.join("\n")}`
    );
  }
};
