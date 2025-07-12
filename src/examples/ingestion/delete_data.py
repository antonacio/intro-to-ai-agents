import logging
import shutil
import sqlite3
from pathlib import Path
from examples.config import (
    vector_store_collection_name,
    vector_store_client,
    vector_store_directory,
)

logger = logging.getLogger(__name__)


def delete_collection_data(collection_name: str):
    """Delete only the specific segment folders for a given collection."""
    logger.info(
        f"Deleting data for collection '{collection_name}' from the vector store ..."
    )

    try:
        # Get the collection to find its UUID
        collection = vector_store_client.get_collection(collection_name)
        collection_id = str(collection.id)
        logger.info(f"Found collection ID: {collection_id}")

        # Find all VECTOR segment IDs (folder names) for this collection
        vector_segment_ids = get_collection_segment_ids(collection_id)

        # Delete the collection from ChromaDB client
        vector_store_client.delete_collection(collection_name)
        logger.info(f"Collection '{collection_name}' deleted from ChromaDB client.")

        # Delete the specific VECTOR segment folders
        for segment_id in vector_segment_ids:
            delete_segment_folder(segment_id)

        logger.info(
            f"Collection '{collection_name}' and its VECTOR segment folders deleted successfully."
        )

    except Exception as e:
        logger.error(f"Error deleting collection '{collection_name}': {e}")
        logger.warning("You may need to perform a manual cleanup.")


def get_collection_segment_ids(collection_id: str) -> list[str]:
    """Query the SQLite database to find VECTOR segment IDs for a given collection."""
    sqlite_path = Path(vector_store_directory) / "chroma.sqlite3"

    if not sqlite_path.exists():
        logger.error(f"SQLite database not found at {sqlite_path}")
        logger.warning("You may need to perform a manual cleanup.")
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
            unknown_segments = []

            for segment_id, segment_type in rows:
                if "vector" in segment_type.lower():
                    vector_segments.append(segment_id)
                    logger.info(
                        f"Found VECTOR segment: {segment_id} ({segment_type}) - has folder"
                    )
                elif "metadata" in segment_type.lower():
                    metadata_segments.append(segment_id)
                    logger.info(
                        f"Found METADATA segment: {segment_id} ({segment_type}) - no folder"
                    )
                else:
                    unknown_segments.append(segment_id)
                    logger.info(
                        f"Found segment of unknown type: {segment_id} ({segment_type}) - no folder"
                    )

            logger.info(
                f"Will delete {len(vector_segments)} VECTOR segment folder(s), "
                f"skipping {len(metadata_segments)} METADATA segment(s)"
            )
            return vector_segments

    except Exception as e:
        logger.error(f"Error querying SQLite database: {e}")
        return []


def delete_segment_folder(segment_id: str):
    """Delete a specific segment UUID folder."""
    vector_store_path = Path(vector_store_directory)

    if not vector_store_path.exists():
        logger.warning("Vector store directory doesn't exist, nothing to clean up.")
        return

    segment_folder = vector_store_path / segment_id

    if segment_folder.exists() and segment_folder.is_dir():
        try:
            # Get folder size before deletion
            folder_size = sum(
                f.stat().st_size for f in segment_folder.glob("**/*") if f.is_file()
            )
            logger.info(
                f"Removing segment folder: {segment_id} (Size: {folder_size / (1024 * 1024):.1f} MB)"
            )
            shutil.rmtree(segment_folder)
            logger.info(f"Successfully removed segment folder: {segment_id}")
        except Exception as e:
            logger.error(f"Error removing segment folder {segment_id}: {e}")
    else:
        logger.warning(
            f"Segment folder {segment_id} not found in vector store directory"
        )


if __name__ == "__main__":
    logger.info("--> Starting the data deletion pipeline ...")

    delete_collection_data(vector_store_collection_name)

    logger.info("--> Data deletion pipeline completed successfully.")
