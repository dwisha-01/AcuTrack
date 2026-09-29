import sqlite3
import numpy as np

def inspect():
    conn = sqlite3.connect("accutrack.db")
    cursor = conn.cursor()
    
    print("=== Tracked Persons ===")
    cursor.execute("SELECT person_id, best_image_path, visit_count, is_flagged_suspicious, created_at FROM tracked_persons")
    persons = cursor.fetchall()
    for p in persons:
        print(f"ID: {p[0]} | Image: {p[1]} | Visits: {p[2]} | Flagged: {p[3]} | Created: {p[4]}")
        
    print("\n=== Person Embeddings ===")
    cursor.execute("SELECT id, person_id, camera_key, length(embedding_data) FROM person_embeddings")
    embs = cursor.fetchall()
    for e in embs:
        print(f"Emb ID: {e[0]} | Person ID: {e[1]} | Camera: {e[2]} | Bytes: {e[3]}")
        
    print("\n=== Sightings Summary ===")
    cursor.execute("SELECT person_id, camera_id, count(*) FROM sightings GROUP BY person_id, camera_id")
    sightings = cursor.fetchall()
    for s in sightings:
        print(f"Person: {s[0]} | Camera DB ID: {s[1]} | Count: {s[2]}")
        
    conn.close()

if __name__ == "__main__":
    inspect()
