const DEFAULT_COV_URL =
  "https://app.covidence.org/reviews/645754/review_studies/select?filter=vote_required_from";

const DEFAULT_CHROME_PLUGIN_ROOT =
  "/Users/liuha/.codex/plugins/cache/openai-bundled/chrome/0.1.7";

export const TAGS = [
  "A1c",
  "Accuracy",
  "Adult",
  "AUC/ROC",
  "Awaiting classification",
  "CDSS",
  "CGM/Glucose",
  "CLIP",
  "CNN",
  "Community/Remote",
  "Comorbidity prediction",
  "Diabetes",
  "Diabetes diagnosis",
  "Diabetes risk prediction",
  "Diabetes treatment/management",
  "Diabetic foot ulcers",
  "Diabetic kideny disease (DKD)",
  "Diabetic Macular Edema (DME)",
  "Diabetic macular ischemia (DMI)",
  "Diabetic nephropathy",
  "Diabetic retinopathy",
  "Diagnosis/Prediction",
  "Diffusion Models",
  "DNN (Deep Neural Network)",
  "ED",
  "EHR/Claims",
  "External Validation",
  "F1",
  "Federated Learning",
  "Foot/Wound Imaging",
  "GAN",
  "GDM",
  "Generative‑Clinician",
  "Generative‑Patient",
  "Genomics/Omics",
  "GNN",
  "Hypo/Hyper events",
  "Hypoglycemia/hyperglycemia",
  "Inpatient",
  "Internal Validation",
  "Labs",
  "LLM",
  "LOS/Readmission",
  "LSTM",
  "Mean Average Precision (mAP)",
  "Multimodal",
  "Ongoing study",
  "Other DL Methods",
  "Outpatient/Primary Care",
  "Patient education/engagement",
  "Paywall",
  "Pediatric",
  "Precision/F1/MCC",
  "Pregnancy/Gestational diabetes",
  "Prospective",
  "RAG",
  "RCT",
  "Real‑world Implementation",
  "Retinal Imaging",
  "Retrospective",
  "Review",
  "RL (Reinforcement Learning)",
  "RMSE",
  "RNN",
  "Sensitivity/Specificity",
  "T1D",
  "T2D",
  "Text / Notes",
  "Time‑in‑Range",
  "Traditional ML",
  "Transformer",
  "Treatment/Management",
  "VAE",
  "ViT",
  "Wearables/Sensors",
];

const TAG_SET = new Set(TAGS);

function normalize(text) {
  return String(text || "")
    .normalize("NFKC")
    .replace(/[‐‑‒–—]/g, "-")
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function has(text, pattern) {
  if (pattern instanceof RegExp) {
    return pattern.test(text);
  }
  return text.includes(normalize(pattern));
}

function any(text, patterns) {
  return patterns.some((pattern) => has(text, pattern));
}

function add(tags, tag, reason) {
  if (!TAG_SET.has(tag)) {
    throw new Error(`Unknown tag emitted by classifier: ${tag}`);
  }
  tags.set(tag, reason);
}

export function classifyStudyTags(study) {
  const tags = new Map();
  const title = normalize(study.title);
  const abstract = normalize(study.abstract);
  const text = normalize(`${study.title} ${study.abstract}`);

  if (any(text, [/\bdiabet(es|ic|ic\b)/, /\bt1d\b/, /\bt2d\b/, /blood glucose/, /\bbgl\b/, /glycemic/, /glycaemic/])) {
    add(tags, "Diabetes", "diabetes or glucose/glycemic context");
  }

  if (any(text, [/hba1c/, /\ba1c\b/, /glycated hemoglobin/, /glycated haemoglobin/])) {
    add(tags, "A1c", "A1c/HbA1c is mentioned");
  }

  if (any(text, [/continuous glucose monitor/, /\bcgm\b/, /blood glucose/, /\bbgl\b/, /glucose level/, /glucose prediction/, /glucose forecast/, /glycemic/, /glycaemic/])) {
    add(tags, "CGM/Glucose", "glucose monitoring or prediction");
  }

  if (any(text, [/photoplethysmography/, /\bppg\b/, /wearable/, /sensor/, /\biot\b/, /smartwatch/, /mobile sensor/])) {
    add(tags, "Wearables/Sensors", "sensor or wearable signal/data");
  }

  if (any(text, [/retin/, /fundus/, /\boct\b/, /angiograph/, /macular/, /ophthalm/])) {
    add(tags, "Retinal Imaging", "retinal or ophthalmic imaging");
  }

  if (any(text, [/diabetic retinopathy/, /\bdr\b/]) && any(text, [/retin/, /fundus/, /ophthalm/])) {
    add(tags, "Diabetic retinopathy", "diabetic retinopathy");
  }

  if (any(text, [/macular edema/, /macular oedema/, /\bdme\b/])) {
    add(tags, "Diabetic Macular Edema (DME)", "diabetic macular edema/oedema");
  }

  if (any(text, [/macular ischemia/, /macular ischaemia/, /\bdmi\b/])) {
    add(tags, "Diabetic macular ischemia (DMI)", "diabetic macular ischemia/ischaemia");
  }

  if (any(text, [/diabetic foot/, /foot ulcer/, /wound/])) {
    add(tags, "Diabetic foot ulcers", "diabetic foot ulcer or wound context");
    add(tags, "Foot/Wound Imaging", "foot or wound imaging/context");
  }

  if (any(text, [/kidney disease/, /\bdkd\b/, /nephropathy/, /renal/])) {
    add(tags, "Diabetic kideny disease (DKD)", "diabetic kidney disease or renal context");
  }

  if (any(text, [/nephropathy/])) {
    add(tags, "Diabetic nephropathy", "diabetic nephropathy");
  }

  if (any(text, [/gestational diabetes/, /\bgdm\b/, /pregnan/])) {
    add(tags, "GDM", "gestational diabetes");
    add(tags, "Pregnancy/Gestational diabetes", "pregnancy or gestational diabetes");
  }

  if (any(text, [/\bt1d\b/, /type 1 diabetes/, /type-1 diabetes/, /type i diabetes/])) {
    add(tags, "T1D", "type 1 diabetes");
  }

  if (any(text, [/\bt2d\b/, /type 2 diabetes/, /type-2 diabetes/, /type ii diabetes/])) {
    add(tags, "T2D", "type 2 diabetes");
  }

  if (any(text, [/diagnos/, /detect/, /classif/, /predict/, /forecast/, /regression/, /risk score/, /risk prediction/])) {
    add(tags, "Diagnosis/Prediction", "diagnosis, detection, prediction, or regression task");
  }

  if (
    any(title, [/diagnos/, /detect/, /classif/, /screen/]) &&
    any(title, [/diabetes mellitus/, /diabetes disease/, /\bdiabetes\b/]) &&
    !any(title, [/retinopathy/, /macular/, /nephropathy/, /kidney/, /foot/])
  ) {
    add(tags, "Diabetes diagnosis", "diabetes diagnosis/detection task");
  }

  if (any(text, [/diabetes risk/, /risk prediction/, /risk score/, /incident diabetes/, /incidence of diabetes/, /prevalence and incidence of diabetes/])) {
    add(tags, "Diabetes risk prediction", "diabetes risk prediction");
  }

  if (any(text, [/comorbid/, /cardiovascular/, /coronary/, /readmission/, /length of stay/, /\blos\b/, /complication prediction/, /predict(?:ing|ion)? (?:of )?(?:diabetes )?complications?/])) {
    add(tags, "Comorbidity prediction", "comorbidity, complication, or related outcome prediction");
  }

  if (any(text, [/diabetes management/, /disease management/, /treatment response/, /anti-vascular endothelial growth factor treatment/, /therapy/, /insulin/, /bolus/, /medication/, /drug selection/, /glucose monitoring/, /monitoring .*glucose/, /glucose .*monitoring/, /blood glucose prediction/, /adherence/, /artificial pancreas/, /dietary assessment/, /dietary intake/, /nutrition/])) {
    add(tags, "Diabetes treatment/management", "diabetes care, monitoring, or treatment management");
    add(tags, "Treatment/Management", "treatment or management context");
  }

  if (any(text, [/clinical decision support/, /\bcdss\b/, /decision support/, /case management/])) {
    add(tags, "CDSS", "clinical decision support");
  }

  if (any(text, [/patient education/, /patient engagement/, /conversational/, /chatbot/, /self-management/])) {
    add(tags, "Patient education/engagement", "patient education or engagement");
  }

  if (any(text, [/electronic health record/, /\behr\b/, /\bemr\b/, /claims/, /administrative claim/])) {
    add(tags, "EHR/Claims", "EHR, EMR, claims, or administrative data");
  }

  if (any(text, [/laborator/, /\blabs?\b/, /medical examination data/])) {
    add(tags, "Labs", "laboratory, biomarker, or medical examination data");
  }

  if (any(text, [/genom/, /genetic/, /gene\b/, /micro.?rna/, /\bmirna\b/, /omics/])) {
    add(tags, "Genomics/Omics", "genomics/omics data");
  }

  if (any(text, [/clinical note/, /medical note/, /free-text/, /textual/, /report analysis/, /\bnlp\b/])) {
    add(tags, "Text / Notes", "text, notes, reports, or NLP");
  }

  if (any(text, [/multimodal/, /multi-modal/, /multi modal/])) {
    add(tags, "Multimodal", "multimodal data or model");
  }

  if (any(text, [/convolutional neural network/, /convolution neural network/, /\bcnn\b/, /\bconvnet\b/, /convolutional recurrent/])) {
    add(tags, "CNN", "CNN/convolutional model");
  }

  if (any(text, [/\blstm\b/, /long short-term memory/, /long short term memory/])) {
    add(tags, "LSTM", "LSTM model");
  }

  if (any(text, [/\brnn\b/, /recurrent neural network/])) {
    add(tags, "RNN", "RNN model");
  }

  if (any(text, [/transformer/, /\bbert\b/, /attention-based/, /attention based/])) {
    add(tags, "Transformer", "transformer or attention model");
  }

  if (any(text, [/vision transformer/, /\bvit\b/])) {
    add(tags, "ViT", "vision transformer");
  }

  if (any(text, [/graph convolution/, /graph neural/, /\bgnn\b/, /\bgcn\b/])) {
    add(tags, "GNN", "graph neural network");
  }

  if (any(text, [/generative adversarial/, /\bgan\b/])) {
    add(tags, "GAN", "GAN model");
  }

  if (any(text, [/variational autoencoder/, /\bvae\b/])) {
    add(tags, "VAE", "VAE model");
  }

  if (any(text, [/diffusion model/, /stable diffusion/])) {
    add(tags, "Diffusion Models", "diffusion model");
  }

  if (any(text, [/\bclip\b/])) {
    add(tags, "CLIP", "CLIP model");
  }

  if (any(text, [/large language model/, /\bllm\b/, /\bgpt\b/])) {
    add(tags, "LLM", "large language model");
  }

  if (any(text, [/retrieval augmented generation/, /\brag\b/])) {
    add(tags, "RAG", "retrieval augmented generation");
  }

  if (any(text, [/reinforcement learning/, /\brl\b/])) {
    add(tags, "RL (Reinforcement Learning)", "reinforcement learning");
  }

  if (any(text, [/federated learning/, /\bfed[a-z-]*\b/])) {
    add(tags, "Federated Learning", "federated learning");
  }

  if (any(text, [/deep neural network/, /\bdnn\b/, /neural network/, /deep learning network/])) {
    add(tags, "DNN (Deep Neural Network)", "neural network/deep neural network");
  }

  if (
    any(text, [/deep learning/, /transfer learning/, /multi.?task learning/, /multitask learning/, /self-supervised/, /capsule network/]) &&
    !any(text, [/convolutional neural network/, /\bcnn\b/, /\blstm\b/, /\brnn\b/, /transformer/, /\bgnn\b/, /\bgan\b/, /\bvae\b/, /diffusion/])
  ) {
    add(tags, "Other DL Methods", "deep learning method without a more specific method tag");
  }

  if (
    any(text, [/machine learning/, /random forest/, /\bsvm\b/, /support vector/, /xgboost/, /gradient boost/, /logistic regression/, /naive bayes/, /decision tree/]) &&
    !any(text, [/deep learning/, /neural network/, /\bcnn\b/, /\blstm\b/, /transformer/])
  ) {
    add(tags, "Traditional ML", "traditional machine learning method");
  }

  if (any(text, [/\baccuracy\b/, /accurate/])) {
    add(tags, "Accuracy", "accuracy metric");
  }

  if (any(text, [/\bauc\b/, /\broc\b/, /receiver operating/])) {
    add(tags, "AUC/ROC", "AUC/ROC metric");
  }

  if (any(text, [/\bf1\b/, /f-1/, /f1-score/])) {
    add(tags, "F1", "F1 metric");
    add(tags, "Precision/F1/MCC", "precision/F1/MCC metric family");
  }

  if (any(text, [/precision/, /\bmcc\b/, /matthews correlation/])) {
    add(tags, "Precision/F1/MCC", "precision or MCC metric");
  }

  if (any(text, [/sensitivity/, /specificity/, /sensitivities/, /specificities/])) {
    add(tags, "Sensitivity/Specificity", "sensitivity/specificity metric");
  }

  if (any(text, [/mean average precision/, /\bmap\b/])) {
    add(tags, "Mean Average Precision (mAP)", "mean average precision metric");
  }

  if (any(text, [/\brmse\b/, /root mean squared error/])) {
    add(tags, "RMSE", "RMSE metric");
  }

  if (any(text, [/external validation/, /externally validat/, /multi.?center validation/, /multicenter validation/])) {
    add(tags, "External Validation", "external validation");
  }

  if (any(text, [/internal validation/, /cross-validation/, /cross validation/])) {
    add(tags, "Internal Validation", "internal validation or cross-validation");
  }

  if (any(text, [/randomi[sz]ed controlled trial/, /\brct\b/, /randomi[sz]ed crossover/, /randomi[sz]ed trial/])) {
    add(tags, "RCT", "randomized trial");
  }

  if (any(text, [/prospective/])) {
    add(tags, "Prospective", "prospective design");
  }

  if (any(text, [/retrospective/])) {
    add(tags, "Retrospective", "retrospective design");
  }

  if (any(text, [/systematic review/, /meta-analysis/, /scoping review/, /realist review/, /review protocol/])) {
    add(tags, "Review", "review article or protocol");
  }

  if (any(text, [/protocol/, /ongoing study/, /study protocol/]) && !any(abstract, [/results?:/, /achieves?/, /found/, /performance/])) {
    add(tags, "Ongoing study", "protocol or ongoing study");
  }

  if (any(text, [/real-world implementation/, /clinical implementation/, /deployment/, /deployed/, /primary care setting/, /community-based/])) {
    add(tags, "Real‑world Implementation", "real-world implementation or deployment");
  }

  if (any(text, [/community/, /remote/, /home care/, /telemedicine/, /mobile health/, /\bmhealth\b/])) {
    add(tags, "Community/Remote", "community, remote, or home-care setting");
  }

  if (any(text, [/outpatient/, /primary care/, /clinic-based/, /clinical setting/])) {
    add(tags, "Outpatient/Primary Care", "outpatient or primary care setting");
  }

  if (any(text, [/inpatient/, /hospitali[sz]ed/, /hospital setting/])) {
    add(tags, "Inpatient", "inpatient setting");
  }

  if (any(text, [/emergency department/, /\bed\b/])) {
    add(tags, "ED", "emergency department setting");
  }

  if (any(text, [/pediatric/, /paediatric/, /children/, /youth/, /adolescen/])) {
    add(tags, "Pediatric", "pediatric/youth population");
  }

  if (any(text, [/adult/, /older adult/])) {
    add(tags, "Adult", "adult population");
  }

  if (any(text, [/hypoglyc/, /hyperglyc/, /hypo.?hyper/])) {
    add(tags, "Hypoglycemia/hyperglycemia", "hypoglycemia/hyperglycemia");
    add(tags, "Hypo/Hyper events", "hypo/hyper events");
  }

  if (any(text, [/time in range/, /time-in-range/, /\btir\b/])) {
    add(tags, "Time‑in‑Range", "time-in-range metric");
  }

  if (any(text, [/length of stay/, /\blos\b/, /readmission/])) {
    add(tags, "LOS/Readmission", "length of stay or readmission");
  }

  return [...tags.entries()].map(([tag, reason]) => ({ tag, reason }));
}

function parseQuotedField(line, fieldName) {
  const prefix = `${fieldName} "`;
  const start = line.indexOf(prefix);
  if (start < 0) return null;
  const body = line.slice(start + prefix.length);
  const end = body.lastIndexOf('"');
  if (end < 0) return null;
  return body.slice(0, end);
}

function selectedStudyIdFromLabel(label) {
  const match = /^Study #(\d+)\s+/.exec(label || "");
  return match ? match[1] : null;
}

function studyLabelFromRegion(region) {
  const match = region.match(/checkbox "([^"]*Study #\d+[^"]*)"/);
  return match ? match[1] : null;
}

function unquoteSnapshotText(value) {
  const trimmed = value.trim();
  if (!trimmed.startsWith('"')) return trimmed;
  try {
    return JSON.parse(trimmed);
  } catch {
    return trimmed.slice(1, trimmed.endsWith('"') ? -1 : undefined);
  }
}

function extractStudyFromRegion(region, fallbackLabel = null) {
  const selectedLabel = studyLabelFromRegion(region) || fallbackLabel;
  const selectedId = selectedStudyIdFromLabel(selectedLabel);
  const lines = region.split("\n");

  let citation = "";
  let title = selectedLabel ? selectedLabel.replace(/^Study #\d+\s+/, "") : "";
  const abstractParts = [];
  const currentTags = [];
  let inAbstract = false;
  let inTags = false;

  for (const line of lines) {
    if (!citation) {
      const citationMatch = line.match(/paragraph:\s+"#\d+ - .*"$/);
      if (citationMatch) {
        citation = unquoteSnapshotText(line.slice(line.indexOf("paragraph:") + "paragraph:".length));
      }
    }

    const heading = parseQuotedField(line, "heading");
    if (heading && line.includes("[level=3]")) {
      title = heading;
    }

    if (line.includes('button "Hide abstract"') || line.includes('button "Show abstract"')) {
      inAbstract = true;
      continue;
    }

    if (inAbstract && (line.includes("- list:") || line.includes('button "Manage full text"'))) {
      inAbstract = false;
    }

    if (line.includes('img "Study tags:"')) {
      inTags = true;
      continue;
    }

    if (inTags && (line.includes('button "Include"') || line.includes('button "Exclude"'))) {
      inTags = false;
    }

    if (inAbstract && line.includes("- paragraph:")) {
      const paragraph = line.slice(line.indexOf("- paragraph:") + "- paragraph:".length);
      const text = unquoteSnapshotText(paragraph);
      if (text && text !== "Abstract") {
        abstractParts.push(text);
      }
    }

    if (inTags && line.includes("- paragraph:")) {
      const paragraph = line.slice(line.indexOf("- paragraph:") + "- paragraph:".length);
      const text = unquoteSnapshotText(paragraph);
      if (TAG_SET.has(text)) {
        currentTags.push(text);
      }
    }
  }

  return {
    selectedId,
    selectedLabel,
    citation,
    title,
    abstract: abstractParts.join("\n\n"),
    currentTags,
  };
}

function regionForStudyLabel(snapshot, selectedLabel) {
  const needle = `checkbox "${selectedLabel}"`;
  const checkboxIndex = snapshot.indexOf(needle);
  if (checkboxIndex < 0) {
    throw new Error(`Selected study checkbox not found in page snapshot: ${selectedLabel}`);
  }

  const regionStart = snapshot.lastIndexOf('\n  - region "File drop zone":', checkboxIndex);
  const start = regionStart >= 0 ? regionStart : checkboxIndex;
  const nextRegion = snapshot.indexOf('\n  - region "File drop zone":', checkboxIndex + needle.length);
  const end = nextRegion >= 0 ? nextRegion : snapshot.length;
  return snapshot.slice(start, end);
}

export function extractSelectedStudyFromSnapshot(snapshot, selectedLabel) {
  return extractStudyFromRegion(regionForStudyLabel(snapshot, selectedLabel), selectedLabel);
}

export function extractVisibleStudiesFromSnapshot(snapshot) {
  const marker = '\n  - region "File drop zone":';
  const studies = [];
  let cursor = snapshot.indexOf(marker);

  while (cursor >= 0) {
    const next = snapshot.indexOf(marker, cursor + marker.length);
    const region = snapshot.slice(cursor, next >= 0 ? next : snapshot.length);
    if (studyLabelFromRegion(region)) {
      studies.push(extractStudyFromRegion(region));
    }
    cursor = next;
  }

  return studies;
}

async function selectedCheckbox(tab) {
  const selected = await tab.playwright.locator('main input[type="checkbox"]:checked').all();
  const studies = [];

  for (const checkbox of selected) {
    const ariaLabel = await checkbox.getAttribute("aria-label", { timeoutMs: 1000 }).catch(() => null);
    if (ariaLabel && ariaLabel.startsWith("Study #")) {
      studies.push({ checkbox, ariaLabel });
    }
  }

  if (studies.length !== 1) {
    throw new Error(`Expected exactly one selected study, found ${studies.length}`);
  }

  return studies[0];
}

async function selectedStudyCheckboxes(tab) {
  const selected = await tab.playwright.locator('main input[type="checkbox"]:checked').all();
  const studies = [];

  for (const checkbox of selected) {
    const ariaLabel = await checkbox.getAttribute("aria-label", { timeoutMs: 1000 }).catch(() => null);
    if (ariaLabel && ariaLabel.startsWith("Study #")) {
      studies.push({ checkbox, ariaLabel });
    }
  }

  return studies;
}

export async function clearSelectedStudies(tab) {
  await closeTagDialogIfOpen(tab);
  const selected = await selectedStudyCheckboxes(tab);
  const labels = selected.map((study) => study.ariaLabel);
  for (const label of labels) {
    const checkbox = tab.playwright.getByRole("checkbox", { name: label, exact: true });
    await checkbox.first().setChecked(false, { timeoutMs: 5000 });
    await tab.playwright.waitForTimeout(100);
  }
  return labels;
}

export async function selectStudyByLabel(tab, label) {
  let lastError = null;

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      await closeTagDialogIfOpen(tab);
      await clearSelectedStudies(tab);
      const checkbox = tab.playwright.getByRole("checkbox", { name: label, exact: true });
      if ((await checkbox.count().catch(() => 0)) !== 1) {
        throw new Error(`Expected one checkbox for ${label}`);
      }
      await checkbox.first().setChecked(true, { timeoutMs: 7000 });
      await tab.playwright.waitForTimeout(300);
      const selected = await selectedStudyCheckboxes(tab);
      if (selected.some((study) => study.ariaLabel === label)) {
        return;
      }
      throw new Error(`Checkbox did not remain selected for ${label}`);
    } catch (error) {
      lastError = error;
      await tab.playwright.waitForTimeout(700 * attempt);
    }
  }

  throw lastError;
}

async function ensureAbstractVisible(tab, selectedLabel) {
  const snapshot = await tab.playwright.domSnapshot();
  const initialStudy = extractSelectedStudyFromSnapshot(snapshot, selectedLabel);
  if (initialStudy.abstract) {
    return initialStudy;
  }

  const studyId = selectedStudyIdFromLabel(selectedLabel);
  const rowLabel = studyId ? `Study #${studyId}` : selectedLabel;
  const showAbstractButtons = await tab.playwright
    .locator(`input[aria-label="${selectedLabel.replace(/"/g, '\\"')}"]`)
    .count()
    .catch(() => 0);

  if (!showAbstractButtons) {
    return initialStudy;
  }

  // The page snapshot gives us the selected card, but this wrapper has no direct
  // DOM traversal API. If the abstract is collapsed, use the first visible
  // "Show abstract" button after selecting only one study; Covidence renders
  // selected studies at the top of the current viewport in this workflow.
  const showAbstract = tab.playwright.getByRole("button", { name: "Show abstract", exact: true });
  if ((await showAbstract.count().catch(() => 0)) > 0) {
    await showAbstract.first().click({ timeoutMs: 5000 });
    await tab.playwright.waitForTimeout(500);
  }

  const refreshed = await tab.playwright.domSnapshot();
  const refreshedStudy = extractSelectedStudyFromSnapshot(refreshed, selectedLabel);
  if (!refreshedStudy.abstract) {
    refreshedStudy.abstract = `(Abstract not visible in DOM for ${rowLabel}; classification used title/citation only.)`;
  }
  return refreshedStudy;
}

async function findCovidenceTab(browser, url) {
  const tabs = await browser.user.openTabs();
  const open = tabs.find((candidate) => (candidate.url || "").includes("/reviews/645754/"));
  if (open) {
    return browser.user.claimTab(open);
  }
  const tab = await browser.tabs.new();
  await tab.goto(url);
  return tab;
}

export async function connectChromeBrowser(options = {}) {
  if (!globalThis.agent) {
    const pluginRoot = options.chromePluginRoot || DEFAULT_CHROME_PLUGIN_ROOT;
    const { setupAtlasRuntime } = await import(`${pluginRoot}/scripts/browser-client.mjs`);
    await setupAtlasRuntime({ globals: globalThis });
  }

  if (!globalThis.browser) {
    globalThis.browser = await agent.browsers.get("extension");
  }

  await browser.nameSession(options.sessionName || "🔎 Covidence auto-tagger");
  return browser;
}

export async function readSelectedStudy(tab) {
  await closeTagDialogIfOpen(tab);
  const { ariaLabel } = await selectedCheckbox(tab);
  return ensureAbstractVisible(tab, ariaLabel);
}

export async function closeTagDialogIfOpen(tab) {
  const closeButton = tab.playwright.getByRole("button", { name: "Close popover", exact: true });
  if ((await closeButton.count().catch(() => 0)) > 0 && (await closeButton.first().isVisible().catch(() => true))) {
    await closeButton.first().click({ timeoutMs: 5000 });
    await tab.playwright.waitForTimeout(300);
  }
}

export async function openTagDialog(tab) {
  const dialog = tab.playwright.getByRole("dialog", { name: /Apply tags to studies/ });
  if ((await dialog.count().catch(() => 0)) > 0 && (await dialog.first().isVisible().catch(() => true))) {
    return;
  }

  const tagButton = tab.playwright.getByRole("button", { name: /Tag \(\d+ stud(?:y|ies)\)/ });
  await tagButton.click({ timeoutMs: 10000 });
  await tab.playwright.waitForTimeout(500);
}

export async function applyTags(tab, tags) {
  await openTagDialog(tab);
  const applied = [];
  const missing = [];

  for (const tag of tags) {
    const checkbox = tab.playwright.getByRole("checkbox", { name: tag, exact: true });
    const count = await checkbox.count().catch(() => 0);
    if (!count) {
      missing.push(tag);
      continue;
    }
    await checkbox.first().setChecked(true, { timeoutMs: 7000 });
    applied.push(tag);
    await tab.playwright.waitForTimeout(100);
  }

  if (applied.length > 0) {
    const applyButton = tab.playwright.getByRole("button", { name: /Apply \d+ tags/ });
    if ((await applyButton.count().catch(() => 0)) > 0) {
      await applyButton.first().click({ timeoutMs: 10000 });
      await tab.playwright.waitForTimeout(1000);
    }
  }

  return { applied, missing };
}

export async function tagSelectedStudy(options = {}) {
  const mode = options.mode || "dry-run";
  const url = options.url || DEFAULT_COV_URL;
  const browser = options.browser || (await connectChromeBrowser(options));
  const tab = options.tab || (await findCovidenceTab(browser, url));

  try {
    await tab.playwright.waitForLoadState({ state: "domcontentloaded", timeoutMs: 10000 });
  } catch {
    // Covidence sometimes keeps network activity open; DOM readiness is enough.
  }

  const study = await readSelectedStudy(tab);
  const tagReasons = classifyStudyTags(study);
  const tags = tagReasons.map((entry) => entry.tag);
  const result = { mode, study, tagReasons, applied: [], missing: [] };

  if (mode === "apply") {
    const applied = await applyTags(tab, tags);
    result.applied = applied.applied;
    result.missing = applied.missing;
  }

  return result;
}

export async function verifiedStudyTags(tab, selectedLabel) {
  await closeTagDialogIfOpen(tab);
  const snapshot = await tab.playwright.domSnapshot();
  return extractSelectedStudyFromSnapshot(snapshot, selectedLabel).currentTags;
}

function xpathStringLiteral(value) {
  if (!value.includes('"')) {
    return `"${value}"`;
  }
  if (!value.includes("'")) {
    return `'${value}'`;
  }
  return `concat(${value
    .split('"')
    .map((part) => `"${part}"`)
    .join(', \'"\', ')})`;
}

function studyIdFromLabel(label) {
  return (label.match(/^Study #(\d+)/) || [null, null])[1];
}

function titleFromStudyLabel(label) {
  return label.replace(/^Study #\d+\s+/, "");
}

export function cardForStudyLabel(tab, label) {
  return tab.playwright.locator(
    `xpath=//input[@aria-label=${xpathStringLiteral(label)}]/ancestor::*[@role='region'][1]`,
  );
}

export async function visibleStudyLabels(tab, options = {}) {
  const limit = options.limit || Infinity;
  const labels = [];
  const checkboxes = await tab.playwright.locator('main input[type="checkbox"][aria-label^="Study #"]').all();

  for (let index = 0; index < checkboxes.length && labels.length < limit; index += 1) {
    const label = await checkboxes[index].getAttribute("aria-label", { timeoutMs: 1000 }).catch(() => null);
    if (label) {
      labels.push(label);
    }
  }

  return labels;
}

export async function currentTagsForStudyLabel(tab, label) {
  const removeTexts = await cardForStudyLabel(tab, label)
    .getByRole("button", { name: /Remove/ })
    .allTextContents({ timeoutMs: 1500 })
    .catch(() => []);

  return removeTexts.map((text) => text.replace(/^Remove\s+/, "").trim()).filter(Boolean);
}

export async function readStudyFromCard(tab, label) {
  const card = cardForStudyLabel(tab, label);
  const showAbstract = card.getByRole("button", { name: "Show abstract", exact: true });
  if ((await showAbstract.count().catch(() => 0)) > 0) {
    await showAbstract.first().click({ timeoutMs: 7000 });
    await tab.playwright.waitForTimeout(500);
  }

  const cardText = await card.innerText({ timeoutMs: 10000 });
  return {
    selectedId: studyIdFromLabel(label),
    selectedLabel: label,
    citation: (cardText.match(/#\d+\s+-\s+[^\n]+/) || [""])[0],
    title: titleFromStudyLabel(label),
    abstract: cardText,
    currentTags: [],
  };
}

export async function tagNextVisibleUntaggedStudies(options = {}) {
  const mode = options.mode || "dry-run";
  const limit = Number.isFinite(options.limit) ? options.limit : 10;
  const labelScanLimit = Number.isFinite(options.labelScanLimit) ? options.labelScanLimit : 160;
  const startIndex = Number.isFinite(options.startIndex) ? options.startIndex : 0;
  const url = options.url || DEFAULT_COV_URL;
  const browser = options.browser || (await connectChromeBrowser(options));
  const tab = options.tab || (await findCovidenceTab(browser, url));
  const results = [];

  try {
    await tab.playwright.waitForLoadState({ state: "domcontentloaded", timeoutMs: 10000 });
  } catch {
    // Covidence sometimes keeps network activity open; DOM readiness is enough.
  }

  await closeTagDialogIfOpen(tab);
  const labels = (await visibleStudyLabels(tab, { limit: labelScanLimit })).slice(startIndex);

  for (const label of labels) {
    if (results.length >= limit) {
      break;
    }

    const existingTags = await currentTagsForStudyLabel(tab, label);
    if (existingTags.length > 0) {
      continue;
    }

    const study = await readStudyFromCard(tab, label);
    const tagReasons = classifyStudyTags(study);
    const tags = tagReasons.map((entry) => entry.tag);
    const result = {
      mode,
      study,
      tagReasons,
      applied: [],
      missing: [],
      verifiedTags: existingTags,
    };

    if (mode === "apply" && tags.length > 0) {
      await selectStudyByLabel(tab, label);
      const applied = await applyTags(tab, tags);
      result.applied = applied.applied;
      result.missing = applied.missing;
      await closeTagDialogIfOpen(tab);
      result.verifiedTags = await currentTagsForStudyLabel(tab, label);
    }

    results.push(result);
  }

  await clearSelectedStudies(tab).catch(() => {});
  return {
    mode,
    requested: limit,
    processed: results.length,
    results,
  };
}

export async function tagVisibleUntaggedStudies(options = {}) {
  const mode = options.mode || "dry-run";
  const limit = Number.isFinite(options.limit) ? options.limit : 10;
  const url = options.url || DEFAULT_COV_URL;
  const browser = options.browser || (await connectChromeBrowser(options));
  const tab = options.tab || (await findCovidenceTab(browser, url));

  try {
    await tab.playwright.waitForLoadState({ state: "domcontentloaded", timeoutMs: 10000 });
  } catch {
    // Covidence sometimes keeps network activity open; DOM readiness is enough.
  }

  await closeTagDialogIfOpen(tab);
  const startingSnapshot = await tab.playwright.domSnapshot();
  const candidates = extractVisibleStudiesFromSnapshot(startingSnapshot)
    .filter((study) => study.selectedLabel && study.currentTags.length === 0)
    .slice(0, limit);
  const results = [];

  for (const candidate of candidates) {
    await selectStudyByLabel(tab, candidate.selectedLabel);
    const study = await readSelectedStudy(tab);
    const tagReasons = classifyStudyTags(study);
    const tags = tagReasons.map((entry) => entry.tag);
    const result = {
      mode,
      study,
      tagReasons,
      applied: [],
      missing: [],
      verifiedTags: [],
    };

    if (mode === "apply" && tags.length > 0) {
      const applied = await applyTags(tab, tags);
      result.applied = applied.applied;
      result.missing = applied.missing;
      result.verifiedTags = await verifiedStudyTags(tab, candidate.selectedLabel);
    }

    results.push(result);
  }

  await clearSelectedStudies(tab);
  return {
    mode,
    requested: limit,
    processed: results.length,
    results,
  };
}

function parseArgs(args) {
  const options = { mode: "dry-run", command: "selected", limit: 10 };
  for (const arg of args) {
    if (arg === "--apply") options.mode = "apply";
    if (arg === "--dry-run") options.mode = "dry-run";
    if (arg === "--batch") options.command = "batch";
    if (arg === "--fast-batch") options.command = "fast-batch";
    if (arg.startsWith("--limit=")) options.limit = Number(arg.slice("--limit=".length));
    if (arg.startsWith("--url=")) options.url = arg.slice("--url=".length);
    if (arg.startsWith("--chrome-plugin-root=")) {
      options.chromePluginRoot = arg.slice("--chrome-plugin-root=".length);
    }
  }
  return options;
}

const invokedPath = globalThis.process?.argv?.[1];

if (invokedPath && import.meta.url === `file://${invokedPath}`) {
  const options = parseArgs(globalThis.process.argv.slice(2));
  const result =
    options.command === "batch"
      ? await tagVisibleUntaggedStudies(options)
      : options.command === "fast-batch"
        ? await tagNextVisibleUntaggedStudies(options)
      : await tagSelectedStudy(options);
  console.log(JSON.stringify(result, null, 2));
}
