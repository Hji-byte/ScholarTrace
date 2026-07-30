import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = "E:/research_agent";
const outDir = `${root}/outputs/019f8e15-6539-7250-ade6-154a630f2651`;
const labeledPath = `${root}/evaluation/results/dense-rrf-k15-top30-chunk-labeled.jsonl`;
const summaryPath = `${root}/evaluation/results/dense-rrf-k15-top30-chunk-labeled-summary.json`;
const rows = (await fs.readFile(labeledPath, "utf8")).trim().split(/\r?\n/).map(JSON.parse);
const summary = JSON.parse(await fs.readFile(summaryPath, "utf8"));
const cleanText = (value) => {
  if (value === null || value === undefined) return "";
  let text = String(value).replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g, "");
  if (text.startsWith("=")) text = `'${text}`;
  return text;
};

const wb = Workbook.create();
const summarySheet = wb.worksheets.add("Summary");
const labelsSheet = wb.worksheets.add("Labels");
summarySheet.showGridLines = false;
labelsSheet.showGridLines = false;

summarySheet.getRange("A1:J1").merge();
summarySheet.getRange("A1").values = [["Dense + RRF Chunk Relevance Annotation"]];
summarySheet.getRange("A1:J1").format = { fill: "#16324F", font: { bold: true, color: "#FFFFFF", size: 16 }, rowHeight: 30 };
summarySheet.getRange("A3:J3").merge();
summarySheet.getRange("A3").values = [["Codex 单标注者的第一轮相关性判断，不等同于独立人工 gold label。可在 Labels 的 manual_label 列修正；final_label 会自动采用修正值。"]];
summarySheet.getRange("A3:J3").format = { fill: "#EAF2F8", font: { color: "#274C77", italic: true }, wrapText: true, rowHeight: 34 };

summarySheet.getRange("A5:D5").values = [["总 chunk", "2 直接证据", "1 部分证据", "0 无关"]];
summarySheet.getRange("A6").values = [[rows.length]];
summarySheet.getRange("B6").formulas = [[`=COUNTIF('Labels'!$F$2:$F$${rows.length + 1},"2")`]];
summarySheet.getRange("C6").formulas = [[`=COUNTIF('Labels'!$F$2:$F$${rows.length + 1},"1")`]];
summarySheet.getRange("D6").formulas = [[`=COUNTIF('Labels'!$F$2:$F$${rows.length + 1},"0")`]];
summarySheet.getRange("A5:D5").format = { fill: "#2A6F97", font: { bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
summarySheet.getRange("A6:D6").format = { fill: "#F7FAFC", font: { bold: true, size: 13 }, horizontalAlignment: "center", numberFormat: "#,##0" };

const headers = ["question_id", "question", "chunks", "label 2", "label 1", "label 0", "P@5", "P@10", "P@20", "P@30", "Auto nDCG@10"];
summarySheet.getRange("A9:K9").values = [headers];
const questions = [...new Map(rows.map(r => [r.question_id, cleanText(r.question)])).entries()];
summarySheet.getRange(`A10:B${9 + questions.length}`).values = questions;
for (let i = 0; i < questions.length; i++) {
  const row = 10 + i;
  const qCell = `$A${row}`;
  summarySheet.getRange(`C${row}`).formulas = [[`=COUNTIF('Labels'!$B$2:$B$${rows.length + 1},${qCell})`]];
  for (const [col, label] of [["D", "2"], ["E", "1"], ["F", "0"]]) {
    summarySheet.getRange(`${col}${row}`).formulas = [[`=COUNTIFS('Labels'!$B$2:$B$${rows.length + 1},${qCell},'Labels'!$F$2:$F$${rows.length + 1},"${label}")`]];
  }
  for (const [col, k] of [["G", 5], ["H", 10], ["I", 20], ["J", 30]]) {
    const base = `'Labels'!$B$2:$B$${rows.length + 1},${qCell},'Labels'!$C$2:$C$${rows.length + 1},"<=${k}"`;
    const numerator = `COUNTIFS(${base},'Labels'!$F$2:$F$${rows.length + 1},"2")+COUNTIFS(${base},'Labels'!$F$2:$F$${rows.length + 1},"1")`;
    summarySheet.getRange(`${col}${row}`).formulas = [[`=(${numerator})/COUNTIFS(${base})`]];
  }
  const pq = summary.per_question.find(x => x.question_id === questions[i][0]);
  summarySheet.getRange(`K${row}`).values = [[pq.ndcg_at_10]];
}
summarySheet.getRange(`A9:K${9 + questions.length}`).format.borders = { preset: "insideHorizontal", style: "thin", color: "#D9E2EC" };
summarySheet.getRange("A9:K9").format = { fill: "#16324F", font: { bold: true, color: "#FFFFFF" }, wrapText: true, horizontalAlignment: "center" };
summarySheet.getRange(`G10:K${9 + questions.length}`).format.numberFormat = "0.0%";
summarySheet.getRange(`A10:A${9 + questions.length}`).format.font = { bold: true, color: "#16324F" };
summarySheet.getRange(`B10:B${9 + questions.length}`).format.wrapText = true;
summarySheet.getRange("A:A").format.columnWidth = 13;
summarySheet.getRange("B:B").format.columnWidth = 58;
summarySheet.getRange("C:F").format.columnWidth = 11;
summarySheet.getRange("G:K").format.columnWidth = 13;
summarySheet.freezePanes.freezeRows(9);

const labelHeaders = ["item_id", "question_id", "rank", "auto_label", "manual_label", "final_label", "paper_title", "page", "rrf_score", "matched_queries", "question", "chunk_text", "rationale", "url"];
labelsSheet.getRange("A1:N1").values = [labelHeaders];
const values = rows.map(r => [cleanText(r.item_id), cleanText(r.question_id), r.rank, r.auto_label, "", null, cleanText(r.title), r.page, r.rrf_score, r.matched_query_count, cleanText(r.question), cleanText(r.text), cleanText(r.auto_rationale), cleanText(r.url)]);
labelsSheet.getRange(`A2:N${rows.length + 1}`).values = values;
labelsSheet.getRange("F2").formulas = [["=IF(E2=\"\",D2,E2)"]];
labelsSheet.getRange(`F2:F${rows.length + 1}`).fillDown();
labelsSheet.getRange(`E2:E${rows.length + 1}`).dataValidation = { rule: { type: "list", values: ["0", "1", "2", "?"] } };
labelsSheet.getRange("A1:N1").format = { fill: "#16324F", font: { bold: true, color: "#FFFFFF" }, wrapText: true, horizontalAlignment: "center", rowHeight: 28 };
labelsSheet.getRange(`A2:N${rows.length + 1}`).format.borders = { preset: "insideHorizontal", style: "thin", color: "#E3E8EE" };
labelsSheet.getRange(`D2:F${rows.length + 1}`).format.horizontalAlignment = "center";
labelsSheet.getRange(`C2:C${rows.length + 1}`).format.numberFormat = "0";
labelsSheet.getRange(`I2:I${rows.length + 1}`).format.numberFormat = "0.000000";
labelsSheet.getRange(`K2:M${rows.length + 1}`).format.wrapText = true;
labelsSheet.getRange(`A2:N${rows.length + 1}`).format.rowHeight = 58;
for (const [range, width] of [["A:A",18],["B:B",12],["C:C",7],["D:F",12],["G:G",42],["H:H",7],["I:I",12],["J:J",12],["K:K",55],["L:L",90],["M:M",35],["N:N",28]]) labelsSheet.getRange(range).format.columnWidth = width;
labelsSheet.getRange(`F2:F${rows.length + 1}`).conditionalFormats.add("cellIs", { operator: "equal", formula: 2, format: { fill: "#D9EAD3", font: { color: "#27632A" } } });
labelsSheet.getRange(`F2:F${rows.length + 1}`).conditionalFormats.add("cellIs", { operator: "equal", formula: 1, format: { fill: "#FFF2CC", font: { color: "#7F6000" } } });
labelsSheet.getRange(`F2:F${rows.length + 1}`).conditionalFormats.add("cellIs", { operator: "equal", formula: 0, format: { fill: "#F4CCCC", font: { color: "#990000" } } });
labelsSheet.tables.add(`A1:N${rows.length + 1}`, true, "ChunkLabelsTable");
labelsSheet.freezePanes.freezeRows(1);
labelsSheet.freezePanes.freezeColumns(3);

await fs.mkdir(outDir, { recursive: true });
const summaryPreview = await wb.render({ sheetName: "Summary", range: "A1:K39", scale: 1.25, format: "png" });
await fs.writeFile(`${outDir}/chunk-annotation-summary-preview.png`, new Uint8Array(await summaryPreview.arrayBuffer()));
const labelsPreview = await wb.render({ sheetName: "Labels", range: "A1:N8", scale: 1, format: "png" });
await fs.writeFile(`${outDir}/chunk-annotation-labels-preview.png`, new Uint8Array(await labelsPreview.arrayBuffer()));
console.log((await wb.inspect({ kind: "table", range: "Summary!A1:K15", include: "values,formulas", tableMaxRows: 15, tableMaxCols: 11 })).ndjson);
console.log((await wb.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 50 }, summary: "formula error scan" })).ndjson);
const output = await SpreadsheetFile.exportXlsx(wb);
await output.save(`${outDir}/dense-rrf-k15-chunk-annotations.xlsx`);
console.log(`${outDir}/dense-rrf-k15-chunk-annotations.xlsx`);
