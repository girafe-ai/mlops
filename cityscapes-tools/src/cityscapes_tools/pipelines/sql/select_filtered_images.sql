SELECT DISTINCT
    images.image_id,
    images.split,
    images.city,
    images.image_path,
    images.annotation_path
FROM images
JOIN objects ON objects.image_id = images.image_id
WHERE objects.label_name = ANY(%s)
ORDER BY images.split, images.city, images.image_id;
