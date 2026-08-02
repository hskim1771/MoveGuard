import { integer, real, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const measurementSessions = sqliteTable("measurement_sessions", {
  id: text("id").primaryKey(),
  startedAt: text("started_at").notNull(),
  endedAt: text("ended_at").notNull(),
  durationSeconds: integer("duration_seconds").notNull(),
  maxPeople: integer("max_people").notNull(),
  samples: integer("samples").notNull(),
  normalSamples: integer("normal_samples").notNull(),
  cautionSamples: integer("caution_samples").notNull(),
  dangerSamples: integer("danger_samples").notNull(),
  averageConfidence: real("average_confidence").notNull(),
  metricMax: text("metric_max").notNull(),
  metricAverage: text("metric_average").notNull().default("{}"),
  metricWarnings: text("metric_warnings").notNull(),
  createdAt: text("created_at").notNull(),
});
