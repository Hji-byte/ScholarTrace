import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.dirname(fileURLToPath(import.meta.url));
const candidatesPath = path.join(outputDir, "candidates.jsonl");
const annotationsPath = path.join(outputDir, "annotations.jsonl");
const workbookPath = path.join(outputDir, "qwen3_no_instruct_search_relevance_annotations.xlsx");

async function readJsonl(filePath) {
  const text = await fs.readFile(filePath, "utf8");
  return text.split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
}

const candidates = await readJsonl(candidatesPath);
const annotations = await readJsonl(annotationsPath);
const questions = await readJsonl(path.resolve("evaluation/datasets/cs_questions_v2.jsonl"));
const questionsById = new Map(questions.map((row) => [row.question_id, row]));
const annotationsById = new Map(annotations.map((row) => [row.item_id, row]));
const rows = candidates.map((candidate) => {
  const annotation = annotationsById.get(candidate.item_id);
  if (!annotation) throw new Error(`Missing annotation: ${candidate.item_id}`);
  const questionMetadata = questionsById.get(candidate.question_id) || {};
  return {
    ...candidate,
    domain: candidate.domain || questionMetadata.domain || "",
    topic: candidate.topic || questionMetadata.topic || "",
    question_type: candidate.question_type || questionMetadata.question_type || "",
    difficulty: candidate.difficulty || questionMetadata.difficulty || "",
    year_from: candidate.year_from || questionMetadata.year_from || null,
    year_to: candidate.year_to || questionMetadata.year_to || null,
    ...annotation,
  };
});

const workbook = Workbook.create();
const readme = workbook.worksheets.add("README");
const summary = workbook.worksheets.add("Summary");
const data = workbook.worksheets.add("Annotations");
for (const sheet of [readme, summary, data]) sheet.showGridLines = false;

const navy = "#17324D";
const blue = "#236B8E";
const paleBlue = "#EAF3F8";
const paleGray = "#F4F6F8";
const text = "#17212B";
const green = "#D9EAD3";
const yellow = "#FFF2CC";
const red = "#F4CCCC";
const gray = "#E7E6E6";

// README
readme.getRange("A1:B1").merge();
readme.getRange("A1").values = [["CS 论文搜索结果相关性标注"]];
readme.getRange("A1:B1").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
readme.getRange("A1:B1").format.rowHeight = 34;
const candidateCounts = new Map();
for (const row of rows) {
  candidateCounts.set(row.question_id, (candidateCounts.get(row.question_id) || 0) + 1);
}
const countValues = [...candidateCounts.values()];
const candidateScale = new Set(countValues).size === 1
  ? `${rows.length} 篇；${candidateCounts.size} 题每题 ${countValues[0]} 篇。`
  : `${rows.length} 篇；各题候选数不完全相同。`;
const readmeRows = [
  ["字段", "说明"],
  ["用途", "评估 research agent 在 30 个计算机科学问题上的论文搜索结果相关性。"],
  ["候选规模", candidateScale],
  ["初标性质", "Codex 基于研究问题、论文标题和摘要进行的模型辅助初标，不等同于经过领域专家复核的人类金标准。"],
  ["2 — Directly relevant", "论文主要贡献直接覆盖问题至少一个明确且实质性的子方向，可作为最终综述的核心证据。"],
  ["1 — Partially relevant", "与问题有实质关联，可提供背景、应用、支持技术或侧面证据，但主要贡献不直接回答问题焦点。"],
  ["0 — Irrelevant", "关键词碰撞、任务或领域不符，不能实质帮助回答问题。"],
  ["? — Unclear", "仅凭标题和摘要不足以判断；应查看全文。该标签从严使用。"],
  ["人工复核", "在 Annotations 的“人工修正”列选择 0、1、2 或 ?；“最终标签”会自动采用人工值，否则沿用 Codex 初标。"],
  ["相关论文定义", "Summary 中的相关率和 Precision@K 将标签 1 与 2 都视为 relevant；标签 2 单独代表核心相关。"],
  ["时间口径", "年份为 arXiv 上传年份；候选由各题配置的上传年份范围过滤。"],
  ["结果来源", `候选来自：${[...new Set(rows.map((row) => row.source_run))].join("；")}。`],
];
readme.getRange(`A3:B${2 + readmeRows.length}`).values = readmeRows;
readme.getRange("A3:B3").format = { fill: blue, font: { bold: true, color: "#FFFFFF" } };
readme.getRange(`A4:A${2 + readmeRows.length}`).format = { fill: paleBlue, font: { bold: true, color: text } };
readme.getRange(`A3:B${2 + readmeRows.length}`).format.wrapText = true;
readme.getRange(`A3:B${2 + readmeRows.length}`).format.borders = { preset: "inside", style: "thin", color: "#D9E1E8" };
readme.getRange("A:A").format.columnWidth = 24;
readme.getRange("B:B").format.columnWidth = 100;
readme.getRange(`A3:B${2 + readmeRows.length}`).format.autofitRows();

// Annotations raw/processed table
const headers = [
  "item_id", "question_id", "domain", "topic", "question", "rank", "source_id",
  "title", "authors", "year", "abstract", "url", "Qwen3 reranker score", "source run",
  "Codex 初标", "初标理由", "人工修正", "最终标签", "人工备注",
];
data.getRange(`A1:S1`).values = [headers];
const values = rows.map((row) => [
  row.item_id,
  row.question_id,
  row.domain,
  row.topic,
  row.question,
  row.rank,
  row.source_id,
  row.title,
  (row.authors || []).join("; "),
  row.year,
  row.abstract,
  row.url,
  row.reranker_score,
  row.source_run,
  String(row.label),
  row.rationale,
  "",
  null,
  "",
]);
data.getRange(`A2:S${rows.length + 1}`).values = values;
data.getRange("R2").formulas = [["=IF(Q2<>\"\",Q2,O2)"]];
data.getRange(`R2:R${rows.length + 1}`).fillDown();
data.getRange(`A1:S${rows.length + 1}`).format.font = { name: "Aptos", size: 10, color: text };
data.getRange("A1:S1").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF", size: 10 },
  verticalAlignment: "center",
  wrapText: true,
};
data.getRange("A1:S1").format.rowHeight = 30;
data.freezePanes.freezeRows(1);
data.freezePanes.freezeColumns(2);
data.getRange(`F2:F${rows.length + 1}`).format.numberFormat = "0";
data.getRange(`J2:J${rows.length + 1}`).format.numberFormat = "0";
data.getRange(`M2:M${rows.length + 1}`).format.numberFormat = "0.0000";
data.getRange(`O2:R${rows.length + 1}`).format.horizontalAlignment = "center";
data.getRange(`Q2:Q${rows.length + 1}`).format.fill = "#FFFBE6";
data.getRange(`S2:S${rows.length + 1}`).format.fill = "#FFFBE6";
data.getRange(`Q2:Q${rows.length + 1}`).dataValidation = { rule: { type: "list", values: ["0", "1", "2", "?"] } };
data.getRange(`R2:R${rows.length + 1}`).conditionalFormats.add("cellIs", { operator: "equal", formula: "2", format: { fill: green, font: { bold: true, color: "#274E13" } } });
data.getRange(`R2:R${rows.length + 1}`).conditionalFormats.add("cellIs", { operator: "equal", formula: "1", format: { fill: yellow, font: { color: "#7F6000" } } });
data.getRange(`R2:R${rows.length + 1}`).conditionalFormats.add("cellIs", { operator: "equal", formula: "0", format: { fill: red, font: { color: "#990000" } } });
data.getRange(`R2:R${rows.length + 1}`).conditionalFormats.add("containsText", { text: "?", format: { fill: gray, font: { color: "#595959" } } });
data.tables.add(`A1:S${rows.length + 1}`, true, "AnnotationsTable").style = "TableStyleMedium2";
const widths = {
  A: 17, B: 12, C: 24, D: 24, E: 52, F: 8, G: 18, H: 55, I: 36, J: 9,
  K: 78, L: 34, M: 12, N: 48, O: 12, P: 36, Q: 12, R: 12, S: 30,
};
for (const [col, width] of Object.entries(widths)) data.getRange(`${col}:${col}`).format.columnWidth = width;
data.getRange(`E2:E${rows.length + 1}`).format.wrapText = true;
data.getRange(`H2:H${rows.length + 1}`).format.wrapText = true;
data.getRange(`K2:K${rows.length + 1}`).format.wrapText = true;
data.getRange(`P2:P${rows.length + 1}`).format.wrapText = true;
data.getRange(`S2:S${rows.length + 1}`).format.wrapText = true;
data.getRange(`A2:S${rows.length + 1}`).format.rowHeight = 54;

// Formula-driven summary
summary.getRange("A1:L1").merge();
summary.getRange("A1").values = [["30 题论文搜索相关性汇总"]];
summary.getRange("A1:L1").format = {
  fill: navy,
  font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
summary.getRange("A1:L1").format.rowHeight = 34;
summary.getRange("A3:L3").values = [[
  "question_id", "question", "候选数", "2 直接相关", "1 部分相关", "0 无关", "? 不确定",
  "相关率", "Precision@5", "Precision@10", "Precision@20", "直接相关率",
]];
const questionRows = [...new Map(rows.map((row) => [row.question_id, row])).values()]
  .sort((a, b) => a.question_id.localeCompare(b.question_id));
summary.getRange(`A4:B${questionRows.length + 3}`).values = questionRows.map((row) => [row.question_id, row.question]);
for (let excelRow = 4; excelRow <= questionRows.length + 3; excelRow++) {
  summary.getRange(`C${excelRow}:L${excelRow}`).formulas = [[
    `=COUNTIF('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow})`,
    `=COUNTIFS('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow},'Annotations'!$R$2:$R$${rows.length + 1},\"2\")`,
    `=COUNTIFS('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow},'Annotations'!$R$2:$R$${rows.length + 1},\"1\")`,
    `=COUNTIFS('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow},'Annotations'!$R$2:$R$${rows.length + 1},\"0\")`,
    `=COUNTIFS('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow},'Annotations'!$R$2:$R$${rows.length + 1},\"~?\")`,
    `=IF(C${excelRow}=0,0,(D${excelRow}+E${excelRow})/C${excelRow})`,
    `=IF(COUNTIFS('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow},'Annotations'!$F$2:$F$${rows.length + 1},\"<=5\")=0,0,(COUNTIFS('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow},'Annotations'!$F$2:$F$${rows.length + 1},\"<=5\",'Annotations'!$R$2:$R$${rows.length + 1},\"2\")+COUNTIFS('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow},'Annotations'!$F$2:$F$${rows.length + 1},\"<=5\",'Annotations'!$R$2:$R$${rows.length + 1},\"1\"))/COUNTIFS('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow},'Annotations'!$F$2:$F$${rows.length + 1},\"<=5\"))`,
    `=IF(COUNTIFS('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow},'Annotations'!$F$2:$F$${rows.length + 1},\"<=10\")=0,0,(COUNTIFS('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow},'Annotations'!$F$2:$F$${rows.length + 1},\"<=10\",'Annotations'!$R$2:$R$${rows.length + 1},\"2\")+COUNTIFS('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow},'Annotations'!$F$2:$F$${rows.length + 1},\"<=10\",'Annotations'!$R$2:$R$${rows.length + 1},\"1\"))/COUNTIFS('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow},'Annotations'!$F$2:$F$${rows.length + 1},\"<=10\"))`,
    `=IF(COUNTIF('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow})=0,0,(COUNTIFS('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow},'Annotations'!$R$2:$R$${rows.length + 1},\"2\")+COUNTIFS('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow},'Annotations'!$R$2:$R$${rows.length + 1},\"1\"))/COUNTIF('Annotations'!$B$2:$B$${rows.length + 1},A${excelRow}))`,
    `=IF(C${excelRow}=0,0,D${excelRow}/C${excelRow})`,
  ]];
}
const totalRow = questionRows.length + 5;
summary.getRange(`A${totalRow}:B${totalRow}`).merge();
summary.getRange(`A${totalRow}`).values = [["总体"]];
summary.getRange(`C${totalRow}:G${totalRow}`).formulas = [[
  `=SUM(C4:C${questionRows.length + 3})`,
  `=SUM(D4:D${questionRows.length + 3})`,
  `=SUM(E4:E${questionRows.length + 3})`,
  `=SUM(F4:F${questionRows.length + 3})`,
  `=SUM(G4:G${questionRows.length + 3})`,
]];
summary.getRange(`H${totalRow}`).formulas = [[`=IF(C${totalRow}=0,0,(D${totalRow}+E${totalRow})/C${totalRow})`]];
summary.getRange(`I${totalRow}:K${totalRow}`).formulas = [[
  `=IF(COUNTIF('Annotations'!$F$2:$F$${rows.length + 1},"<=5")=0,0,(COUNTIFS('Annotations'!$F$2:$F$${rows.length + 1},"<=5",'Annotations'!$R$2:$R$${rows.length + 1},"2")+COUNTIFS('Annotations'!$F$2:$F$${rows.length + 1},"<=5",'Annotations'!$R$2:$R$${rows.length + 1},"1"))/COUNTIF('Annotations'!$F$2:$F$${rows.length + 1},"<=5"))`,
  `=IF(COUNTIF('Annotations'!$F$2:$F$${rows.length + 1},"<=10")=0,0,(COUNTIFS('Annotations'!$F$2:$F$${rows.length + 1},"<=10",'Annotations'!$R$2:$R$${rows.length + 1},"2")+COUNTIFS('Annotations'!$F$2:$F$${rows.length + 1},"<=10",'Annotations'!$R$2:$R$${rows.length + 1},"1"))/COUNTIF('Annotations'!$F$2:$F$${rows.length + 1},"<=10"))`,
  `=IF(C${totalRow}=0,0,(D${totalRow}+E${totalRow})/C${totalRow})`,
]];
summary.getRange(`L${totalRow}`).formulas = [[`=IF(C${totalRow}=0,0,D${totalRow}/C${totalRow})`]];
summary.getRange("A3:L3").format = { fill: blue, font: { bold: true, color: "#FFFFFF" }, wrapText: true };
summary.getRange(`A${totalRow}:L${totalRow}`).format = { fill: paleBlue, font: { bold: true, color: text } };
summary.getRange(`C4:G${totalRow}`).format.numberFormat = "0";
summary.getRange(`H4:L${totalRow}`).format.numberFormat = "0.0%";
summary.getRange(`A3:L${totalRow}`).format.borders = { preset: "inside", style: "thin", color: "#D9E1E8" };
summary.getRange(`B4:B${questionRows.length + 3}`).format.wrapText = true;
summary.getRange(`A4:L${questionRows.length + 3}`).format.rowHeight = 42;
summary.freezePanes.freezeRows(3);
summary.getRange("A:A").format.columnWidth = 13;
summary.getRange("B:B").format.columnWidth = 72;
summary.getRange("C:G").format.columnWidth = 13;
summary.getRange("H:L").format.columnWidth = 16;

await fs.mkdir(outputDir, { recursive: true });
const checks = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:L12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 12,
});
console.log(checks.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "formula error scan",
});
console.log(errors.ndjson);
for (const [sheetName, range, fileName] of [
  ["README", "A1:B14", "preview-readme.png"],
  ["Summary", `A1:L${totalRow}`, "preview-summary.png"],
  ["Annotations", "A1:S18", "preview-annotations.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(outputDir, fileName), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(workbookPath);
console.log(`Saved ${workbookPath}`);
