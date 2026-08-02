CREATE TABLE `measurement_sessions` (
	`id` text PRIMARY KEY NOT NULL,
	`started_at` text NOT NULL,
	`ended_at` text NOT NULL,
	`duration_seconds` integer NOT NULL,
	`max_people` integer NOT NULL,
	`samples` integer NOT NULL,
	`normal_samples` integer NOT NULL,
	`caution_samples` integer NOT NULL,
	`danger_samples` integer NOT NULL,
	`average_confidence` real NOT NULL,
	`metric_max` text NOT NULL,
	`metric_warnings` text NOT NULL,
	`created_at` text NOT NULL
);
