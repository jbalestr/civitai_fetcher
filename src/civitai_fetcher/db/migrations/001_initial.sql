-- Initial schema.
--
-- Normalised out of the old per-image JSON blob: model/resource info that
-- used to repeat on every image row now lives once per model/resource.
-- `meta` is split three ways:
--   - the handful of fields worth filtering/sorting on -> plain columns on `images`
--   - civitaiResources (checkpoint/LoRA usage, can be many per image) -> image_resources
--   - the full original payload (comfy workflow graphs and all, can be 100KB+
--     for Krea/Anima-style tools) -> raw_meta, compressed
-- `stats`/`reactionScore` are the one genuinely time-varying part of an
-- image record, so they get their own append-only history table instead of
-- being overwritten in place -- refreshing an image's reactions later never
-- loses the earlier reading.

CREATE TABLE models (
    modelId     INTEGER PRIMARY KEY,
    modelName   TEXT,
    modelUrl    TEXT
);

CREATE TABLE resources (
    modelVersionId    INTEGER PRIMARY KEY,
    name              TEXT,
    versionName       TEXT,
    creatorUsername   TEXT,
    resource_type     TEXT  -- 'checkpoint' | 'lora' | null (unresolved)
);

CREATE TABLE images (
    imageId          INTEGER PRIMARY KEY,
    modelId          INTEGER REFERENCES models(modelId),
    modelVersionId   INTEGER REFERENCES resources(modelVersionId),  -- checkpoint used, if known
    imageUrl         TEXT,
    posterUsername   TEXT,
    postId           INTEGER,
    postUrl          TEXT,
    width            INTEGER,
    height           INTEGER,
    media_type       TEXT CHECK (media_type IN ('image', 'video', 'audio')),
    createdAt        TEXT,     -- ISO string, as returned by the API
    nsfwLevel        INTEGER,
    prompt           TEXT,
    negativePrompt   TEXT,
    sampler          TEXT,
    steps            INTEGER,
    cfgScale         REAL,
    file_path        TEXT,     -- local path if/when media bytes are ever downloaded; null until then
    first_seen_at    TEXT NOT NULL,  -- when this row was first inserted (for "new since" queries)
    enriched_at      TEXT      -- set once resource/creator-name enrichment has run; null = needs enrichment
);

-- An image can use several resources at once (one checkpoint + N LoRAs).
CREATE TABLE image_resources (
    imageId          INTEGER NOT NULL REFERENCES images(imageId),
    modelVersionId   INTEGER NOT NULL REFERENCES resources(modelVersionId),
    weight           REAL,
    resource_type    TEXT,  -- 'checkpoint' | 'lora', as reported on this specific image
    PRIMARY KEY (imageId, modelVersionId)
);

-- Append-only: every refresh adds a row rather than overwriting, so trend
-- analysis over reaction growth stays possible.
CREATE TABLE image_stats (
    imageId        INTEGER NOT NULL REFERENCES images(imageId),
    fetched_at     TEXT NOT NULL,
    likeCount      INTEGER,
    heartCount     INTEGER,
    laughCount     INTEGER,
    cryCount       INTEGER,
    commentCount   INTEGER,
    reactionScore  INTEGER,
    PRIMARY KEY (imageId, fetched_at)
);

-- Full original meta payload, zlib-compressed, kept for anything not worth
-- promoting to a real column (comfy workflow graphs, tool-specific fields).
CREATE TABLE raw_meta (
    imageId       INTEGER PRIMARY KEY REFERENCES images(imageId),
    meta_blob     BLOB,       -- zlib-compressed JSON
    raw_size      INTEGER,    -- uncompressed byte size, for visibility into compression ratio
    compressed_size INTEGER
);

CREATE TABLE schema_version (
    version    INTEGER NOT NULL
);
INSERT INTO schema_version (version) VALUES (1);

-- Partial index: only covers rows still needing enrichment, so it stays
-- cheap regardless of total table size.
CREATE INDEX idx_images_unenriched ON images (enriched_at) WHERE enriched_at IS NULL;
CREATE INDEX idx_images_first_seen ON images (first_seen_at);
CREATE INDEX idx_images_model ON images (modelId);
CREATE INDEX idx_images_created ON images (createdAt);
CREATE INDEX idx_image_stats_imageId ON image_stats (imageId);
