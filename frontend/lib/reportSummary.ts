type GenericRecord = Record<string, unknown>;

function asRecord(value: unknown): GenericRecord | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as GenericRecord)
    : null;
}

function text(value: unknown): string | undefined {
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed || undefined;
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return String(value);
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      const extracted = text(item);
      if (extracted) return extracted;
    }
    return undefined;
  }
  const record = asRecord(value);
  if (!record) return undefined;
  return (
    text(record.summary) ||
    text(record.title) ||
    text(record.note) ||
    text(record.text) ||
    text(record.item)
  );
}

function firstText(...values: unknown[]): string | undefined {
  for (const value of values) {
    const extracted = text(value);
    if (extracted) return extracted;
  }
  return undefined;
}

export function extractReportSummary(
  role: "student" | "professor",
  report: unknown
): string | undefined {
  const safe = asRecord(report);
  if (!safe) return undefined;

  const priorityIssue = asRecord(safe.priority_issue);

  return role === "student"
    ? firstText(
        safe.summary,
        safe.overall_judgment,
        priorityIssue?.title,
        safe.confidence_explanation
      )
    : firstText(
        safe.summary,
        safe.evaluator_overview,
        safe.feedback_explanation,
        safe.confidence_explanation,
        safe.moderation_notes
      );
}

export function summarizeForCard(
  role: "student" | "professor",
  report: unknown,
  limit = 220
): string {
  const summary = extractReportSummary(role, report) || "No summary available for this report yet.";
  return summary.length > limit ? `${summary.slice(0, limit)}...` : summary;
}
