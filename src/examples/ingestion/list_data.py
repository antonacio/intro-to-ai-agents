import logging
import sqlite3
from pathlib import Path
from examples.config import (
    vector_store_client,
    vector_store_directory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


def get_collection_segment_ids(collection_id: str) -> list[str]:
    """Query the SQLite database to find VECTOR segment IDs for a given collection."""
    sqlite_path = Path(vector_store_directory) / "chroma.sqlite3"

    if not sqlite_path.exists():
        logger.warning(f"  SQLite database not found at {sqlite_path}")
        return []

    try:
        with sqlite3.connect(str(sqlite_path)) as conn:
            cursor = conn.cursor()
            # Query segments table to find all segments for this collection
            cursor.execute(
                "SELECT id, type FROM segments WHERE collection = ?", (collection_id,)
            )
            rows = cursor.fetchall()

            # Separate VECTOR and METADATA segments
            vector_segments = []
            metadata_segments = []

            for segment_id, segment_type in rows:
                if "vector" in segment_type.lower():
                    vector_segments.append(segment_id)
                    logger.info(
                        f"  Found VECTOR segment: {segment_id} ({segment_type}) - has folder"
                    )
                elif "metadata" in segment_type.lower():
                    metadata_segments.append(segment_id)
                    logger.info(
                        f"  Found METADATA segment: {segment_id} ({segment_type}) - no folder"
                    )
                else:
                    logger.info(
                        f"  Found segment of unknown type: {segment_id} ({segment_type}) - no folder"
                    )

            logger.info(
                f"  Collection has {len(vector_segments)} VECTOR segment(s) and {len(metadata_segments)} METADATA segment(s)"
            )
            return vector_segments

    except Exception as e:
        logger.error(f"  Error querying SQLite database: {e}")
        return []


def list_collections_and_folders():
    """Utility function to list collections and their corresponding segment folders."""
    logger.info("  Listing collections and their segment folders...")

    try:
        # List all collections
        collections = vector_store_client.list_collections()
        logger.info(f"  Found {len(collections)} collections:")

        total_size_all = 0
        for collection in collections:
            try:
                collection_id = str(collection.id)
                logger.info(
                    f"  Collection: '{collection.name}' -> ID: {collection_id} | metadata: {collection.metadata}"
                )

                # Get VECTOR segment IDs for this collection (the ones that create folders)
                vector_segment_ids = get_collection_segment_ids(collection_id)

                # Check if corresponding folders exist
                vector_store_path = Path(vector_store_directory)
                total_size = 0
                for segment_id in vector_segment_ids:
                    segment_folder = vector_store_path / segment_id
                    if segment_folder.exists():
                        folder_size = sum(
                            f.stat().st_size
                            for f in segment_folder.glob("**/*")
                            if f.is_file()
                        )
                        total_size += folder_size
                        logger.info(
                            f"     VECTOR segment folder: {segment_id} (Size: {folder_size / (1024 * 1024):.1f} MB)"
                        )
                    else:
                        logger.info(f"     VECTOR segment folder missing: {segment_id}")

                logger.info(f"  Total folder size: {total_size / (1024 * 1024):.1f} MB")
                total_size_all += total_size

            except Exception as e:
                logger.error(f"  Error accessing collection '{collection.name}': {e}")

        logger.info(
            f"  Total size across all collections: {total_size_all / (1024 * 1024):.1f} MB"
        )

    except Exception as e:
        logger.error(f"  Error listing collections: {e}")


if __name__ == "__main__":
    logger.info("--> Starting the data listing pipeline ...")

    list_collections_and_folders()

    logger.info("--> Data listing pipeline completed successfully.")
