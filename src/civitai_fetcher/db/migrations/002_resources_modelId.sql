-- Adds modelId to resources, so a checkpoint's owning model can be found
-- even for creator-scope fetches (images.modelId is deliberately NULL
-- there -- see fetch.py -- but civitaiResources entries still carry the
-- checkpoint's modelId alongside its modelVersionId; this was previously
-- being discarded on the way into `resources`).
ALTER TABLE resources ADD COLUMN modelId INTEGER REFERENCES models(modelId);

CREATE INDEX idx_resources_modelId ON resources (modelId);

INSERT INTO schema_version (version) VALUES (2);