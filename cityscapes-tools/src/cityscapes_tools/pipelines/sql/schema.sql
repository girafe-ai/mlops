CREATE TABLE IF NOT EXISTS images (
    image_id TEXT PRIMARY KEY,
    split TEXT NOT NULL,
    city TEXT NOT NULL,
    image_path TEXT NOT NULL,
    annotation_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS objects (
    image_id TEXT NOT NULL REFERENCES images(image_id) ON DELETE CASCADE,
    object_index INTEGER NOT NULL,
    label_name TEXT NOT NULL,
    polygon JSONB NOT NULL,
    PRIMARY KEY (image_id, object_index)
);

CREATE INDEX IF NOT EXISTS objects_label_name_index ON objects (label_name);
