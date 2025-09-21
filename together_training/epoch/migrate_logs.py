#!/usr/bin/env python3
"""
Log Migration Utility for Inspect AI Logs

This script migrates logs from the old hardcoded format (eval-21-9-1) 
to the new dynamic format (DD-MM-v1) to maintain compatibility.

Usage:
    python migrate_logs.py --old-format eval-21-9-1 --new-format 21-09-v1
    python migrate_logs.py --auto-detect  # Auto-detect and migrate all old format logs
    python migrate_logs.py --list-old     # List all old format log directories
"""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class LogMigrator:
    """Handles migration of Inspect AI logs between different directory formats."""
    
    def __init__(self, log_base_dir: str = "../../logs"):
        """
        Initialize log migrator.
        
        Args:
            log_base_dir: Base directory for logs
        """
        self.log_base_dir = Path(log_base_dir)
        self.log_base_dir.mkdir(parents=True, exist_ok=True)
    
    def find_old_format_logs(self) -> List[str]:
        """
        Find all directories that match old log formats.
        
        Returns:
            List of old format directory names
        """
        old_formats = []
        
        if not self.log_base_dir.exists():
            return old_formats
        
        for item in self.log_base_dir.iterdir():
            if item.is_dir():
                # Check for old formats like "eval-21-9-1", "eval-15-01-2", etc.
                if item.name.startswith("eval-") and len(item.name.split("-")) >= 4:
                    old_formats.append(item.name)
                # Also check for other hardcoded formats
                elif item.name in ["eval", "evaluation", "test"]:
                    old_formats.append(item.name)
        
        return sorted(old_formats)
    
    def detect_new_format(self, old_format: str) -> str:
        """
        Auto-detect appropriate new format from old format.
        
        Args:
            old_format: Old format directory name (e.g., "eval-21-9-1")
            
        Returns:
            New format directory name (e.g., "21-09-v1")
        """
        # Try to extract date from old format
        if old_format.startswith("eval-"):
            parts = old_format.replace("eval-", "").split("-")
            if len(parts) >= 3:
                try:
                    day = parts[0].zfill(2)
                    month = parts[1].zfill(2)
                    version = f"v{parts[2]}"
                    return f"{day}-{month}-{version}"
                except (ValueError, IndexError):
                    pass
        
        # Fallback to current date + version
        current_date = datetime.now()
        return f"{current_date.strftime('%d-%m')}-v1"
    
    def get_log_structure(self, log_dir: Path) -> Dict:
        """
        Analyze the structure of a log directory.
        
        Args:
            log_dir: Path to log directory
            
        Returns:
            Dictionary describing the log structure
        """
        structure = {
            "total_files": 0,
            "total_size": 0,
            "folds": {},
            "epochs": set(),
            "file_types": set()
        }
        
        if not log_dir.exists():
            return structure
        
        for root, dirs, files in os.walk(log_dir):
            root_path = Path(root)
            rel_path = root_path.relative_to(log_dir)
            
            for file in files:
                file_path = root_path / file
                if file_path.exists():
                    structure["total_files"] += 1
                    structure["total_size"] += file_path.stat().st_size
                    structure["file_types"].add(file_path.suffix)
                    
                    # Extract fold and epoch info from path
                    path_parts = rel_path.parts
                    if len(path_parts) >= 1:
                        fold_name = path_parts[0]
                        if fold_name not in structure["folds"]:
                            structure["folds"][fold_name] = {"files": 0, "epochs": set()}
                        structure["folds"][fold_name]["files"] += 1
                        
                        if len(path_parts) >= 2:
                            epoch_info = path_parts[1]
                            structure["folds"][fold_name]["epochs"].add(epoch_info)
                            structure["epochs"].add(epoch_info)
        
        # Convert sets to lists for JSON serialization
        structure["epochs"] = sorted(list(structure["epochs"]))
        structure["file_types"] = sorted(list(structure["file_types"]))
        for fold_info in structure["folds"].values():
            fold_info["epochs"] = sorted(list(fold_info["epochs"]))
        
        return structure
    
    def copy_logs(self, old_format: str, new_format: str, dry_run: bool = False) -> Dict:
        """
        Copy logs from old format to new format, skipping existing files.
        
        Args:
            old_format: Old format directory name
            new_format: New format directory name
            dry_run: If True, only simulate the copy operation
            
        Returns:
            Dictionary with copy operation results
        """
        old_dir = self.log_base_dir / old_format
        new_dir = self.log_base_dir / new_format
        
        result = {
            "source": str(old_dir),
            "destination": str(new_dir),
            "files_copied": 0,
            "files_skipped": 0,
            "total_size": 0,
            "errors": [],
            "dry_run": dry_run
        }
        
        if not old_dir.exists():
            result["errors"].append(f"Source directory does not exist: {old_dir}")
            return result
        
        try:
            if dry_run:
                print(f"[DRY RUN] Would copy {old_dir} -> {new_dir}")
                # Calculate what would be copied/skipped
                for root, dirs, files in os.walk(old_dir):
                    root_path = Path(root)
                    rel_path = root_path.relative_to(old_dir)
                    dest_root = new_dir / rel_path
                    
                    for file in files:
                        src_file = root_path / file
                        dest_file = dest_root / file
                        
                        if dest_file.exists():
                            result["files_skipped"] += 1
                            print(f"  [SKIP] {dest_file} (already exists)")
                        else:
                            result["files_copied"] += 1
                            result["total_size"] += src_file.stat().st_size
                            print(f"  [COPY] {src_file} -> {dest_file}")
            else:
                print(f"Copying {old_dir} -> {new_dir} (skipping existing files)")
                
                # Create destination directory
                new_dir.mkdir(parents=True, exist_ok=True)
                
                # Recursively copy files, skipping existing ones
                for root, dirs, files in os.walk(old_dir):
                    root_path = Path(root)
                    rel_path = root_path.relative_to(old_dir)
                    dest_root = new_dir / rel_path
                    
                    # Create destination subdirectories
                    dest_root.mkdir(parents=True, exist_ok=True)
                    
                    for file in files:
                        src_file = root_path / file
                        dest_file = dest_root / file
                        
                        if dest_file.exists():
                            result["files_skipped"] += 1
                            print(f"  ⏭️  Skipping {rel_path / file} (already exists)")
                        else:
                            try:
                                shutil.copy2(src_file, dest_file)
                                result["files_copied"] += 1
                                result["total_size"] += src_file.stat().st_size
                                print(f"  ✅ Copied {rel_path / file}")
                            except Exception as e:
                                result["errors"].append(f"Failed to copy {src_file}: {str(e)}")
                                print(f"  ❌ Failed to copy {rel_path / file}: {str(e)}")
            
        except Exception as e:
            result["errors"].append(f"Copy operation failed: {str(e)}")
        
        return result
    
    def create_migration_record(self, migrations: List[Dict]) -> None:
        """
        Create a record of migration operations.
        
        Args:
            migrations: List of migration operation results
        """
        record_file = self.log_base_dir / "migration_record.json"
        
        record_data = {
            "migration_date": datetime.now().isoformat(),
            "migrations": migrations,
            "total_operations": len(migrations),
            "successful_operations": len([m for m in migrations if not m["errors"]])
        }
        
        with open(record_file, 'w') as f:
            json.dump(record_data, f, indent=2)
        
        print(f"Migration record saved to: {record_file}")


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate Inspect AI logs between directory formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python migrate_logs.py --old-format eval-21-9-1 --new-format 21-09-v1
  python migrate_logs.py --auto-detect --dry-run
  python migrate_logs.py --list-old
        """
    )
    
    parser.add_argument(
        "--old-format",
        type=str,
        help="Old format directory name (e.g., eval-21-9-1)"
    )
    
    parser.add_argument(
        "--new-format", 
        type=str,
        help="New format directory name (e.g., 21-09-v1)"
    )
    
    parser.add_argument(
        "--auto-detect",
        action="store_true",
        help="Auto-detect and migrate all old format logs"
    )
    
    parser.add_argument(
        "--list-old",
        action="store_true", 
        help="List all old format log directories"
    )
    
    parser.add_argument(
        "--log-base-dir",
        type=str,
        default="../../logs",
        help="Base directory for logs (default: ../../logs)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the migration without actually copying files"
    )
    
    args = parser.parse_args()
    
    # Initialize migrator
    migrator = LogMigrator(args.log_base_dir)
    
    # Handle list command
    if args.list_old:
        print(f"\n{'=' * 60}")
        print(f"OLD FORMAT LOG DIRECTORIES")
        print(f"{'=' * 60}")
        
        old_formats = migrator.find_old_format_logs()
        if not old_formats:
            print("No old format log directories found.")
            return
        
        for old_format in old_formats:
            old_dir = migrator.log_base_dir / old_format
            structure = migrator.get_log_structure(old_dir)
            suggested_new = migrator.detect_new_format(old_format)
            
            print(f"\n📁 {old_format}")
            print(f"   Files: {structure['total_files']}")
            print(f"   Size: {structure['total_size']:,} bytes")
            print(f"   Folds: {len(structure['folds'])}")
            print(f"   Epochs: {structure['epochs']}")
            print(f"   Suggested new format: {suggested_new}")
        return
    
    migrations = []
    
    # Handle auto-detect migration
    if args.auto_detect:
        print(f"\n{'=' * 60}")
        print(f"AUTO-DETECTING OLD FORMAT LOGS")
        print(f"{'=' * 60}")
        
        old_formats = migrator.find_old_format_logs()
        if not old_formats:
            print("No old format log directories found.")
            return
        
        print(f"Found {len(old_formats)} old format directories:")
        for old_format in old_formats:
            suggested_new = migrator.detect_new_format(old_format)
            print(f"  {old_format} -> {suggested_new}")
        
        if not args.dry_run:
            response = input(f"\nProceed with migration? (y/N): ")
            if response.lower() != 'y':
                print("Migration cancelled.")
                return
        
        for old_format in old_formats:
            new_format = migrator.detect_new_format(old_format)
            print(f"\n🔄 Migrating {old_format} -> {new_format}")
            
            result = migrator.copy_logs(old_format, new_format, args.dry_run)
            migrations.append(result)
            
            if result["errors"]:
                print(f"❌ Migration failed: {'; '.join(result['errors'])}")
            else:
                print(f"✅ Migration successful: {result['files_copied']} files copied, {result['files_skipped']} skipped, {result['total_size']:,} bytes")
    
    # Handle specific migration
    elif args.old_format and args.new_format:
        print(f"\n{'=' * 60}")
        print(f"MIGRATING SPECIFIC LOG DIRECTORY")
        print(f"{'=' * 60}")
        
        print(f"Source: {args.old_format}")
        print(f"Destination: {args.new_format}")
        
        # Show structure of source directory
        old_dir = migrator.log_base_dir / args.old_format
        structure = migrator.get_log_structure(old_dir)
        print(f"Source contains: {structure['total_files']} files, {structure['total_size']:,} bytes")
        print(f"Folds: {list(structure['folds'].keys())}")
        print(f"Epochs: {structure['epochs']}")
        
        if not args.dry_run:
            response = input(f"\nProceed with migration? (y/N): ")
            if response.lower() != 'y':
                print("Migration cancelled.")
                return
        
        result = migrator.copy_logs(args.old_format, args.new_format, args.dry_run)
        migrations.append(result)
        
        if result["errors"]:
            print(f"❌ Migration failed: {'; '.join(result['errors'])}")
        else:
            print(f"✅ Migration successful: {result['files_copied']} files copied, {result['files_skipped']} skipped, {result['total_size']:,} bytes")
    
    else:
        parser.error("Must specify either --auto-detect, --list-old, or both --old-format and --new-format")
    
    # Create migration record
    if migrations and not args.dry_run:
        migrator.create_migration_record(migrations)
    
    # Summary
    if migrations:
        successful = len([m for m in migrations if not m["errors"]])
        total_files = sum(m["files_copied"] for m in migrations)
        total_size = sum(m["total_size"] for m in migrations)
        
        print(f"\n{'=' * 60}")
        print(f"MIGRATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Operations: {successful}/{len(migrations)} successful")
        print(f"Files processed: {total_files:,}")
        print(f"Total size: {total_size:,} bytes")
        
        if args.dry_run:
            print("\n⚠️  This was a dry run - no files were actually copied")


if __name__ == "__main__":
    main()