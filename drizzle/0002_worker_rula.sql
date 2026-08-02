ALTER TABLE `measurement_sessions` ADD `worker_name` text DEFAULT '미지정 작업자' NOT NULL;
ALTER TABLE `measurement_sessions` ADD `job_name` text DEFAULT '미지정 작업' NOT NULL;
ALTER TABLE `measurement_sessions` ADD `rula_score` integer;
