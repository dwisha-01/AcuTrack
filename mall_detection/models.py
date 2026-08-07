import os
import json
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, Boolean, 
    DateTime, ForeignKey, Text, LargeBinary, Index
)
from sqlalchemy.orm import relationship, sessionmaker, scoped_session, declarative_base

# Define the local SQLite database file
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///accutrack.db")

# Create engine. 'check_same_thread=False' is required for SQLite in multi-threaded apps
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(DATABASE_URL, connect_args=connect_args)

# Create a thread-safe scoped session factory
db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))

Base = declarative_base()
Base.query = db_session.query_property()

# ==========================================
# 1. CAMERA MODEL
# ==========================================
class Camera(Base):
    __tablename__ = 'cameras'
    
    id = Column(Integer, primary_key=True)
    camera_key = Column(String(50), unique=True, nullable=False) # e.g. 'cam1', 'cam2', 'live'
    label = Column(String(100), nullable=False)                  # e.g. 'Camera 1'
    description = Column(Text, nullable=True)                    # Detailed description
    source_url = Column(String(255), nullable=False)              # File path or RTSP stream URL
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    zones = relationship("Zone", back_populates="camera", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="camera")
    snapshots = relationship("OccupancySnapshot", back_populates="camera")
    histories = relationship("TrackedPersonHistory", back_populates="camera")
    sightings = relationship("Sighting", back_populates="camera")

# ==========================================
# 2. ZONE MODEL
# ==========================================
class Zone(Base):
    __tablename__ = 'zones'
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey('cameras.id', ondelete='CASCADE'), nullable=False)
    zone_name = Column(String(100), nullable=False)       # e.g., 'Zone A'
    coords_json = Column(Text, nullable=False)             # JSON representation of bounding box/polygon
    color_rgb = Column(String(20), nullable=False)        # Hex or RGB e.g. '74,144,217'
    capacity = Column(Integer, default=10)                # Capacity threshold for overcrowding
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    camera = relationship("Camera", back_populates="zones")
    dwell_times = relationship("DwellTime", back_populates="zone", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="zone")
    snapshots = relationship("OccupancySnapshot", back_populates="zone")

# ==========================================
# 3. TRACKED PERSON MODEL (Cross-Camera)
# ==========================================
class TrackedPerson(Base):
    __tablename__ = 'tracked_persons'
    
    person_id = Column(Integer, primary_key=True)         # Matches the cross-camera global_id
    best_image_path = Column(String(255), nullable=True)  # File path to the highest-quality crop snapshot
    first_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    total_dwell = Column(Integer, default=0)              # Cached aggregated dwell time (seconds)
    visit_count = Column(Integer, default=1)              # Number of separate camera sightings
    is_flagged_suspicious = Column(Boolean, default=False)
    flagged_reason = Column(Text, nullable=True)
    manually_flagged = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    embeddings = relationship("PersonEmbedding", back_populates="person", cascade="all, delete-orphan")
    sightings = relationship("Sighting", back_populates="person", cascade="all, delete-orphan")
    dwell_times = relationship("DwellTime", back_populates="person", cascade="all, delete-orphan")
    histories = relationship("TrackedPersonHistory", back_populates="person", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="person")

# ==========================================
# 4. PERSON RE-ID EMBEDDINGS MODEL
# ==========================================
class PersonEmbedding(Base):
    __tablename__ = 'person_embeddings'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('tracked_persons.person_id', ondelete='CASCADE'), nullable=False)
    embedding_data = Column(LargeBinary, nullable=False)  # Stored as serialized float32 NumPy arrays (BLOB)
    camera_key = Column(String(50), nullable=False)       # Camera angle from which it was extracted
    confidence = Column(Float, default=1.0)              # Quality/confidence index of tracking/extraction
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    person = relationship("TrackedPerson", back_populates="embeddings")

Index('idx_embedding_person', PersonEmbedding.person_id)

# ==========================================
# 5. SIGHTINGS LOG MODEL
# ==========================================
class Sighting(Base):
    __tablename__ = 'sightings'
    
    sighting_id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('tracked_persons.person_id', ondelete='CASCADE'), nullable=False)
    camera_id = Column(Integer, ForeignKey('cameras.id', ondelete='CASCADE'), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    confidence = Column(Float, nullable=False)            # YOLO/StrongSORT tracking confidence score
    zone = Column(String(100), nullable=True)              # Name of active zone (if any) e.g., 'Zone A'
    snapshot_path = Column(String(255), nullable=False)    # File path of the cropped snapshot image
    
    # Relationships
    person = relationship("TrackedPerson", back_populates="sightings")
    camera = relationship("Camera", back_populates="sightings")

Index('idx_sightings_person_timestamp', Sighting.person_id, Sighting.timestamp.desc())

# ==========================================
# 6. DWELL TIMES LOG MODEL
# ==========================================
class DwellTime(Base):
    __tablename__ = 'dwell_times'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('tracked_persons.person_id', ondelete='CASCADE'), nullable=False)
    zone_id = Column(Integer, ForeignKey('zones.id', ondelete='CASCADE'), nullable=False)
    entry_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)            # None/Null if currently inside the zone
    dwell_duration_seconds = Column(Integer, nullable=True) # Set when they leave the zone

    # Relationships
    person = relationship("TrackedPerson", back_populates="dwell_times")
    zone = relationship("Zone", back_populates="dwell_times")

# Create Indexing for performance
Index('idx_dwell_person_id', DwellTime.person_id)
Index('idx_dwell_zone_id', DwellTime.zone_id)

# ==========================================
# 7. DETAILED TRAJECTORY HISTORY (Optional)
# ==========================================
class TrackedPersonHistory(Base):
    __tablename__ = 'tracked_persons_history'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('tracked_persons.person_id', ondelete='CASCADE'), nullable=False)
    camera_id = Column(Integer, ForeignKey('cameras.id', ondelete='CASCADE'), nullable=False)
    x_coord = Column(Integer, nullable=False)             # Box center-x
    y_coord = Column(Integer, nullable=False)             # Box center-y
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    person = relationship("TrackedPerson", back_populates="histories")
    camera = relationship("Camera", back_populates="histories")

Index('idx_history_person_time', TrackedPersonHistory.person_id, TrackedPersonHistory.timestamp)

# ==========================================
# 8. OCCUPANCY SNAPSHOTS (For footfall analysis charts)
# ==========================================
class OccupancySnapshot(Base):
    __tablename__ = 'occupancy_snapshots'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    camera_id = Column(Integer, ForeignKey('cameras.id', ondelete='CASCADE'), nullable=False)
    zone_id = Column(Integer, ForeignKey('zones.id', ondelete='CASCADE'), nullable=False)
    person_count = Column(Integer, nullable=False)        # Count at that instant
    footfall_increment = Column(Integer, default=0)       # New arrivals during this interval

    # Relationships
    camera = relationship("Camera", back_populates="snapshots")
    zone = relationship("Zone", back_populates="snapshots")

Index('idx_occupancy_timestamp', OccupancySnapshot.timestamp)

# ==========================================
# 9. SECURITY & ANOMALY ALERTS MODEL
# ==========================================
class Alert(Base):
    __tablename__ = 'alerts'
    
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey('tracked_persons.person_id', ondelete='SET NULL'), nullable=True)
    camera_id = Column(Integer, ForeignKey('cameras.id', ondelete='CASCADE'), nullable=False)
    zone_id = Column(Integer, ForeignKey('zones.id', ondelete='SET NULL'), nullable=True)
    alert_type = Column(String(50), nullable=False)       # LOITERING, OVERCROWDING, SUSPICIOUS_MOVEMENT
    anomaly_score = Column(Float, nullable=True)          # LSTM Autoencoder MSE score
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default="PENDING")         # PENDING, RESOLVED, FALSE_POSITIVE

    # Relationships
    person = relationship("TrackedPerson", back_populates="alerts")
    camera = relationship("Camera", back_populates="alerts")
    zone = relationship("Zone", back_populates="alerts")


# ==========================================
# INITIALIZATION HELPERS
# ==========================================
def init_db(video_sources_config=None, zones_config=None):
    """
    Creates database tables if they do not exist and populates initial 
    camera/zone settings from the app's configuration.
    """
    Base.metadata.create_all(bind=engine)
    
    session = db_session()
    try:
        # 1. Seed/Sync Cameras from configuration
        if video_sources_config:
            for key, info in video_sources_config.items():
                cam = session.query(Camera).filter(Camera.camera_key == key).first()
                if not cam:
                    print(f"Database: Adding missing camera config '{key}'...")
                    cam = Camera(
                        camera_key=key,
                        label=info.get("label", key),
                        description=info.get("desc", ""),
                        source_url=str(info.get("file", ""))
                    )
                    session.add(cam)
            session.commit()
            
        # 2. Seed/Sync default Zones from configuration
        if zones_config:
            cameras_map = {c.camera_key: c.id for c in session.query(Camera).all()}
            for zone_name, info in zones_config.items():
                for cam_key, cam_id in cameras_map.items():
                    existing_zone = session.query(Zone).filter(
                        Zone.camera_id == cam_id,
                        Zone.zone_name == zone_name
                    ).first()
                    if not existing_zone:
                        print(f"Database: Adding missing zone '{zone_name}' for camera '{cam_key}'...")
                        coords_str = json.dumps(list(info.get("coords", ())))
                        color_str = ",".join(map(str, info.get("color", (74, 144, 217))))
                        
                        capacity_val = 10
                        if "Zone A" in zone_name: capacity_val = 8
                        elif "Zone B" in zone_name: capacity_val = 10
                        elif "Zone C" in zone_name: capacity_val = 8
                        
                        zone = Zone(
                            camera_id=cam_id,
                            zone_name=zone_name,
                            coords_json=coords_str,
                            color_rgb=color_str,
                            capacity=capacity_val
                        )
                        session.add(zone)
            session.commit()
    except Exception as e:
        session.rollback()
        print(f"Database Initialization Error: {e}")
    finally:
        session.close()

def shutdown_session(exception=None):
    """Removes current database session at thread cleanup/app teardown."""
    db_session.remove()
