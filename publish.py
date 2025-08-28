#!/usr/bin/env python3

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Create new Cfn artifacts bucket if not already existing
Build artifacts
Upload artifacts to S3 bucket for deployment with CloudFormation
"""

import concurrent.futures
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import boto3
from botocore.exceptions import ClientError
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


class IDPPublisher:
    def __init__(self, verbose=False):
        self.console = Console()
        self.verbose = verbose
        self.bucket_basename = None
        self.prefix = None
        self.region = None
        self.acl = None
        self.bucket = None
        self.prefix_and_version = None
        self.version = None
        self.build_errors = []  # Track build errors for verbose reporting
        self.public_sample_udop_model = ""
        self.public = False
        self.main_template = "idp-main.yaml"
        self.use_container_flag = ""
        self.stat_cmd = None
        self.s3_client = None
        self.cf_client = None
        self._print_lock = threading.Lock()  # Thread-safe printing
        self.skip_validation = False  # Flag to skip lambda validation

    def log_verbose(self, message, style="dim"):
        """Log verbose messages if verbose mode is enabled"""
        if self.verbose:
            self.console.print(f"[{style}]{message}[/{style}]")

    def log_error_details(self, component, error_output):
        """Log detailed error information and store for summary"""
        error_info = {"component": component, "error": error_output}
        self.build_errors.append(error_info)

        if self.verbose:
            self.console.print(f"[red]❌ {component} build failed:[/red]")
            self.console.print(f"[red]{error_output}[/red]")
        else:
            self.console.print(
                f"[red]❌ {component} build failed (use --verbose for details)[/red]"
            )

    def print_error_summary(self):
        """Print summary of all build errors"""
        if not self.build_errors:
            return

        self.console.print("\n[red]❌ Build Error Summary:[/red]")
        for i, error_info in enumerate(self.build_errors, 1):
            self.console.print(f"\n[red]{i}. {error_info['component']}:[/red]")
            if self.verbose:
                self.console.print(f"[red]{error_info['error']}[/red]")
            else:
                # Show first few lines of error for non-verbose mode
                error_lines = error_info["error"].strip().split("\n")
                preview_lines = error_lines[:3]  # Show first 3 lines
                for line in preview_lines:
                    self.console.print(f"[red]  {line}[/red]")
                if len(error_lines) > 3:
                    self.console.print(
                        f"[dim]  ... ({len(error_lines) - 3} more lines, use --verbose for full output)[/dim]"
                    )

    def thread_safe_print(self, message):
        """Thread-safe print method using Rich console"""
        with self._print_lock:
            self.console.print(message)

    def print_usage(self):
        """Print usage information with Rich formatting"""
        self.console.print("\n[bold cyan]Usage:[/bold cyan]")
        self.console.print(
            "  python3 publish.py <cfn_bucket_basename> <cfn_prefix> <region> [public] [--max-workers N] [--verbose]"
        )

        self.console.print("\n[bold cyan]Parameters:[/bold cyan]")
        self.console.print(
            "  [yellow]<cfn_bucket_basename>[/yellow]: Base name for the CloudFormation artifacts bucket"
        )
        self.console.print("  [yellow]<cfn_prefix>[/yellow]: S3 prefix for artifacts")
        self.console.print("  [yellow]<region>[/yellow]: AWS region for deployment")
        self.console.print(
            "  [yellow][public][/yellow]: Optional. If 'public', artifacts will be made publicly readable"
        )
        self.console.print(
            "  [yellow][--max-workers N][/yellow]: Optional. Maximum number of concurrent workers (default: auto-detect)"
        )
        self.console.print(
            "                     Use 1 for sequential processing, higher numbers for more concurrency"
        )
        self.console.print(
            "  [yellow][--verbose, -v][/yellow]: Optional. Enable verbose output for debugging"
        )

    def check_parameters(self, args):
        """Check and validate input parameters"""
        if len(args) < 3:
            self.console.print("[red]Error: Missing required parameters[/red]")
            self.print_usage()
            sys.exit(1)

        # Parse arguments
        self.bucket_basename = args[0]
        self.prefix = args[1].rstrip("/")  # Remove trailing slash
        self.region = args[2]

        # Default values
        self.public = False
        self.acl = "bucket-owner-full-control"
        self.max_workers = None  # Auto-detect

        # Parse optional arguments
        remaining_args = args[3:]
        i = 0
        while i < len(remaining_args):
            arg = remaining_args[i]

            if arg.lower() == "public":
                self.public = True
                self.acl = "public-read"
                self.console.print(
                    "[green]Published S3 artifacts will be accessible by public.[/green]"
                )
            elif arg == "--max-workers":
                if i + 1 >= len(remaining_args):
                    self.console.print(
                        "[red]Error: --max-workers requires a number[/red]"
                    )
                    self.print_usage()
                    sys.exit(1)
                try:
                    self.max_workers = int(remaining_args[i + 1])
                    if self.max_workers < 1:
                        self.console.print(
                            "[red]Error: --max-workers must be at least 1[/red]"
                        )
                        sys.exit(1)
                    self.console.print(
                        f"[green]Using {self.max_workers} concurrent workers[/green]"
                    )
                    i += 1  # Skip the next argument (the number)
                except ValueError:
                    self.console.print(
                        "[red]Error: --max-workers must be followed by a valid number[/red]"
                    )
                    self.print_usage()
                    sys.exit(1)
            elif arg in ["--verbose", "-v"]:
                # Verbose flag is already handled by Typer, just acknowledge it here
                pass
            else:
                self.console.print(
                    f"[yellow]Warning: Unknown argument '{arg}' ignored[/yellow]"
                )

            i += 1

        if not self.public:
            self.console.print(
                "[yellow]Published S3 artifacts will NOT be accessible by public.[/yellow]"
            )

    def setup_environment(self):
        """Set up environment variables and derived values"""
        os.environ["AWS_DEFAULT_REGION"] = self.region

        # Initialize AWS clients
        self.s3_client = boto3.client("s3", region_name=self.region)
        self.cf_client = boto3.client("cloudformation", region_name=self.region)

        # Read version
        try:
            with open("./VERSION", "r") as f:
                self.version = f.read().strip()
        except FileNotFoundError:
            self.console.print("[red]Error: VERSION file not found[/red]")
            sys.exit(1)

        self.prefix_and_version = f"{self.prefix}/{self.version}"
        self.bucket = f"{self.bucket_basename}-{self.region}"

        # Set platform-specific commands
        if platform.machine() == "x86_64":
            self.stat_cmd = "stat --format='%Y'"
        else:
            self.stat_cmd = "stat -f %m"

        # Set UDOP model path based on region
        if self.region == "us-east-1":
            self.public_sample_udop_model = "s3://aws-ml-blog-us-east-1/artifacts/genai-idp/udop-finetuning/rvl-cdip/model.tar.gz"
        elif self.region == "us-west-2":
            self.public_sample_udop_model = "s3://aws-ml-blog-us-west-2/artifacts/genai-idp/udop-finetuning/rvl-cdip/model.tar.gz"
        else:
            self.public_sample_udop_model = ""

    def check_prerequisites(self):
        """Check for required commands and versions"""
        # Check required commands
        required_commands = ["aws", "sam"]
        for cmd in required_commands:
            if not shutil.which(cmd):
                self.console.print(
                    f"[red]Error: {cmd} is required but not installed[/red]"
                )
                sys.exit(1)

        # Check SAM version
        try:
            result = subprocess.run(
                ["sam", "--version"], capture_output=True, text=True, check=True
            )
            sam_version = result.stdout.split()[3]  # Extract version from output
            min_sam_version = "1.129.0"
            if self.version_compare(sam_version, min_sam_version) < 0:
                self.console.print(
                    f"[red]Error: sam version >= {min_sam_version} is required. (Installed version is {sam_version})[/red]"
                )
                self.console.print(
                    "[yellow]Install: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/manage-sam-cli-versions.html[/yellow]"
                )
                sys.exit(1)
        except subprocess.CalledProcessError:
            self.console.print("[red]Error: Could not determine SAM version[/red]")
            sys.exit(1)

        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        min_python_version = "3.12"
        if self.version_compare(python_version, min_python_version) < 0:
            self.console.print(
                f"[red]Error: Python version >= {min_python_version} is required. (Installed version is {python_version})[/red]"
            )
            sys.exit(1)

    def ensure_aws_sam_directory(self):
        """Ensure .aws-sam directory exists"""
        aws_sam_dir = ".aws-sam"

        if not os.path.exists(aws_sam_dir):
            self.console.print(
                "[yellow].aws-sam directory not found. Creating it...[/yellow]"
            )
            os.makedirs(aws_sam_dir, exist_ok=True)
            self.console.print(
                "[green]✅ Successfully created .aws-sam directory[/green]"
            )
        else:
            if self.verbose:
                self.console.print("[dim].aws-sam directory already exists[/dim]")

    def version_compare(self, version1, version2):
        """Compare two version strings. Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2"""

        def normalize(v):
            return [int(x) for x in v.split(".")]

        v1_parts = normalize(version1)
        v2_parts = normalize(version2)

        # Pad shorter version with zeros
        max_len = max(len(v1_parts), len(v2_parts))
        v1_parts.extend([0] * (max_len - len(v1_parts)))
        v2_parts.extend([0] * (max_len - len(v2_parts)))

        for i in range(max_len):
            if v1_parts[i] < v2_parts[i]:
                return -1
            elif v1_parts[i] > v2_parts[i]:
                return 1
        return 0

    def setup_artifacts_bucket(self):
        """Create bucket if necessary"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            self.console.print(f"[green]Using existing bucket: {self.bucket}[/green]")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                self.console.print(
                    f"[yellow]Creating s3 bucket: {self.bucket}[/yellow]"
                )
                try:
                    if self.region == "us-east-1":
                        self.s3_client.create_bucket(Bucket=self.bucket)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket,
                            CreateBucketConfiguration={
                                "LocationConstraint": self.region
                            },
                        )

                    # Enable versioning
                    self.s3_client.put_bucket_versioning(
                        Bucket=self.bucket,
                        VersioningConfiguration={"Status": "Enabled"},
                    )
                except ClientError as create_error:
                    self.console.print(
                        f"[red]Failed to create bucket: {create_error}[/red]"
                    )
                    sys.exit(1)
            else:
                self.console.print(f"[red]Error accessing bucket: {e}[/red]")
                sys.exit(1)

    def clean_temp_files(self, verbose_output=True):
        """Clean temporary files in ./lib"""
        if verbose_output:
            self.console.print("[yellow]Delete temp files in ./lib[/yellow]")
        lib_dir = "./lib"
        if os.path.exists(lib_dir):
            for root, dirs, files in os.walk(lib_dir):
                # Remove __pycache__ directories
                for dir_name in dirs[:]:
                    if dir_name == "__pycache__":
                        shutil.rmtree(os.path.join(root, dir_name))
                        dirs.remove(dir_name)

                # Remove .pyc files
                for file_name in files:
                    if file_name.endswith(".pyc"):
                        os.remove(os.path.join(root, file_name))

    def get_file_checksum(self, file_path):
        """Get SHA256 checksum of a file"""
        if not os.path.exists(file_path):
            return ""

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def get_directory_checksum(self, directory):
        """Get combined checksum of all files in a directory, excluding development artifacts"""
        if not os.path.exists(directory):
            return ""

        # Define patterns to exclude from checksum calculation
        exclude_dirs = {
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "build",
            "dist",
            ".aws-sam",
            "node_modules",
            ".git",
            ".vscode",
            ".idea",
            "test-reports",  # Exclude test report directories
        }

        exclude_file_patterns = {
            ".checksum",
            ".build_checksum",
            "lib/.checksum",
            ".pyc",
            ".pyo",
            ".pyd",
            ".so",
            ".egg-info",
            ".coverage",
            ".DS_Store",
            "Thumbs.db",
            "coverage.xml",  # Coverage reports
            "test-results.xml",  # Test result reports
            ".gitkeep",  # Git placeholder files
        }

        exclude_file_suffixes = (
            ".pyc",
            ".pyo",
            ".pyd",
            ".so",
            ".coverage",
            ".log",  # Log files
        )
        exclude_dir_suffixes = (".egg-info",)

        def should_exclude_dir(dir_name):
            """Check if directory should be excluded from checksum"""
            if dir_name in exclude_dirs:
                return True
            if any(dir_name.endswith(suffix) for suffix in exclude_dir_suffixes):
                return True
            # Exclude test directories for library checksum only
            if "lib" in directory and (
                dir_name == "tests" or dir_name.startswith("test_")
            ):
                return True
            return False

        def should_exclude_file(file_name):
            """Check if file should be excluded from checksum"""
            if file_name in exclude_file_patterns:
                return True
            if any(file_name.endswith(suffix) for suffix in exclude_file_suffixes):
                return True
            # Exclude test files for library checksum only
            if "lib" in directory and (
                file_name.startswith("test_")
                or file_name.endswith("_test.py")
                or file_name == "nodeids"  # pytest cache files
                or file_name == "lastfailed"  # pytest cache files
                or file_name
                in ["coverage.xml", "test-results.xml"]  # specific test report files
            ):
                return True
            return False

        checksums = []
        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories in-place to prevent os.walk from descending into them
            dirs[:] = [d for d in dirs if not should_exclude_dir(d)]

            # Sort to ensure consistent ordering
            dirs.sort()
            files.sort()

            for file in files:
                if not should_exclude_file(file):
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        checksums.append(self.get_file_checksum(file_path))

        # Combine all checksums
        combined = "".join(checksums)
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_checksum(self, *paths):
        """Get combined checksum for multiple paths (files or directories)"""
        checksums = []
        for path in paths:
            if os.path.isfile(path):
                checksums.append(self.get_file_checksum(path))
            elif os.path.isdir(path):
                checksums.append(self.get_directory_checksum(path))

        combined = "".join(checksums)
        return hashlib.sha256(combined.encode()).hexdigest()

    def set_checksum(self, directory):
        """Set checksum for a directory in .checksum file"""
        checksum = self.get_checksum(directory)
        checksum_file = os.path.join(directory, ".checksum")
        with open(checksum_file, "w") as f:
            f.write(checksum)

    def get_stored_checksum(self, directory):
        """Get stored checksum from .checksum file"""
        checksum_file = os.path.join(directory, ".checksum")
        if os.path.exists(checksum_file):
            with open(checksum_file, "r") as f:
                return f.read().strip()
        return ""

    def needs_rebuild(self, *paths):
        """Check if any of the paths have changed since last build"""
        # Use get_component_checksum for consistency with component dependency system
        current_checksum = self.get_component_checksum(*paths)
        current_checksum = self.get_checksum(*paths)

        # For single directory, check its stored checksum
        if len(paths) == 1 and os.path.isdir(paths[0]):
            stored_checksum = self.get_stored_checksum(paths[0])
            return current_checksum != stored_checksum

        # For multiple paths or files, use a global checksum file
        checksum_file = ".build_checksum"
        if os.path.exists(checksum_file):
            with open(checksum_file, "r") as f:
                stored_checksum = f.read().strip()
            return current_checksum != stored_checksum

        return True

    def set_file_checksum(self, path):
        """Set checksum for files in global checksum file"""
        checksum = self.get_checksum(path)
        with open(".build_checksum", "w") as f:
            f.write(checksum)

    def show_build_optimization_info(self):
        """Show information about build optimizations"""
        self.console.print("\n[bold cyan]Build Optimizations Enabled:[/bold cyan]")
        self.console.print("  ✅ SAM Build Caching (--cached)")
        self.console.print("  ✅ Template-level Concurrency (multiple templates)")
        self.console.print("  ✅ Smart Checksum (excludes dev artifacts)")
        self.console.print("  ✅ Selective Cache Clearing (only when lib changes)")

        # Check if cache directories exist
        cache_info = []
        for pattern in ["pattern-1", "pattern-2", "pattern-3"]:
            pattern_dir = f"./patterns/{pattern}"
            if os.path.exists(pattern_dir):
                cache_dir = os.path.join(pattern_dir, ".aws-sam", "cache")
                if os.path.exists(cache_dir):
                    cache_info.append(f"  📁 {pattern}: Cache exists")
                else:
                    cache_info.append(f"  📁 {pattern}: No cache (first build)")

        if cache_info:
            self.console.print("\n[bold cyan]Cache Status:[/bold cyan]")
            for info in cache_info:
                self.console.print(info)

    def ensure_idp_common_library_ready(self):
        """Ensure idp_common library is properly built and ready for use by Lambda functions."""
        self.console.print(
            "[cyan]Ensuring idp_common library is ready for Lambda builds...[/cyan]"
        )

        lib_dir = Path("./lib/idp_common_pkg")
        if not lib_dir.exists():
            self.console.print(
                "[red]Error: idp_common library directory not found[/red]"
            )
            sys.exit(1)

        # Use a lock file to prevent concurrent library builds
        lock_file = lib_dir / ".build_lock"

        # Wait for any existing build to complete
        max_wait = 300  # 5 minutes
        wait_time = 0
        while lock_file.exists() and wait_time < max_wait:
            self.log_verbose("Waiting for concurrent library build to complete...")
            time.sleep(1)
            wait_time += 1

        if lock_file.exists():
            self.console.print("[red]Timeout waiting for library build lock[/red]")
            sys.exit(1)

        try:
            # Create lock file
            lock_file.touch()

            # Build the library to ensure it's ready
            result = subprocess.run(
                [sys.executable, "setup.py", "build"],
                cwd=lib_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                self.console.print(
                    f"[red]Error building idp_common library: {result.stderr}[/red]"
                )
                sys.exit(1)

            # Also create a wheel for better pip compatibility
            result = subprocess.run(
                [sys.executable, "setup.py", "bdist_wheel"],
                cwd=lib_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode != 0:
                self.log_verbose(f"Warning: Could not create wheel: {result.stderr}")

            self.log_verbose("idp_common library build completed successfully")

        except subprocess.TimeoutExpired:
            self.console.print("[red]Timeout building idp_common library[/red]")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"[red]Error building idp_common library: {e}[/red]")
            sys.exit(1)
        finally:
            # Remove lock file
            if lock_file.exists():
                lock_file.unlink()

    def clean_lib(self, verbose_output=True):
        """Clean previous build artifacts in ./lib"""
        if verbose_output:
            self.console.print(
                "[yellow]Cleaning previous build artifacts in ./lib/idp_common_pkg[/yellow]"
            )
        lib_pkg_dir = "./lib/idp_common_pkg"

        # Remove build directories
        for dir_name in ["build", "dist"]:
            dir_path = os.path.join(lib_pkg_dir, dir_name)
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

        # Remove egg-info directories
        for item in os.listdir(lib_pkg_dir):
            if item.endswith(".egg-info"):
                shutil.rmtree(os.path.join(lib_pkg_dir, item))

        # Remove __pycache__ directories and .pyc files
        for root, dirs, files in os.walk(lib_pkg_dir):
            for dir_name in dirs[:]:
                if dir_name == "__pycache__":
                    shutil.rmtree(os.path.join(root, dir_name))
                    dirs.remove(dir_name)

            for file_name in files:
                if file_name.endswith(".pyc"):
                    os.remove(os.path.join(root, file_name))

    def clean_and_build(self, template_path):
        """Clean previous build artifacts and run sam build"""
        dir_path = os.path.dirname(template_path)

        # If dir_path is empty (template.yaml in current directory), use current directory
        if not dir_path:
            dir_path = "."

        # Only clean build artifacts if lib changed, otherwise preserve cache
        lib_changed = hasattr(self, "_lib_changed") and self._lib_changed
        aws_sam_dir = os.path.join(dir_path, ".aws-sam")

        if lib_changed and os.path.exists(aws_sam_dir):
            build_dir = os.path.join(aws_sam_dir, "build")
            cache_dir = os.path.join(aws_sam_dir, "cache")

            if os.path.exists(build_dir):
                print(f"Cleaning build artifacts in {build_dir} (lib changed)")
                shutil.rmtree(build_dir)

            if os.path.exists(cache_dir):
                print(f"Clearing SAM cache in {cache_dir} (lib changed)")
                shutil.rmtree(cache_dir)

        # Check if this template has idp_common dependencies
        has_idp_common_deps = self._check_template_for_idp_common_deps(template_path)

        # Run sam build with cwd parameter (thread-safe)
        abs_dir_path = os.path.abspath(dir_path)
        
        # For patterns and options, use template.yaml directly
        # For main template, use the full template path
        if dir_path == "." or template_path == "template.yaml":
            template_filename = "template.yaml"
        else:
            template_filename = "template.yaml"  # Always use template.yaml for consistency
        
        # Ensure we're using the correct template file
        template_full_path = os.path.join(abs_dir_path, template_filename)
        if not os.path.exists(template_full_path):
            print(f"Error: Template file not found: {template_full_path}")
            sys.exit(1)
        
        cmd = [
            "sam",
            "build",
            "--template-file",
            template_filename,
            "--cached",
        ]

        # Only add --parallel if no idp_common dependencies to prevent race conditions
        if not has_idp_common_deps:
            cmd.append("--parallel")

        if self.use_container_flag and self.use_container_flag.strip():
            cmd.append(self.use_container_flag)

        result = subprocess.run(cmd, cwd=abs_dir_path)
        if result.returncode != 0:
            # If cached build fails, try without cache
            print(f"Cached build failed, retrying without cache for {template_filename}")
            cmd_no_cache = [c for c in cmd if c != "--cached"]
            result = subprocess.run(cmd_no_cache, cwd=abs_dir_path)
            if result.returncode != 0:
                print("Error running sam build")
                sys.exit(1)

    def _check_template_for_idp_common_deps(self, template_path):
        """Check if a template has Lambda functions with idp_common dependencies."""
        template_dir = Path(template_path).parent

        # For main template, check src/lambda directory
        if template_path == "template.yaml":
            src_dir = Path("src/lambda")
        else:
            # For pattern/option templates, check src directory
            src_dir = template_dir / "src"

        if src_dir.exists():
            for func_dir in src_dir.iterdir():
                if func_dir.is_dir():
                    requirements_file = func_dir / "requirements.txt"
                    if requirements_file.exists():
                        try:
                            content = requirements_file.read_text(encoding="utf-8")
                            if "idp_common" in content:
                                return True
                        except Exception:
                            continue
        return False

    def build_and_package_template(self, directory, force_rebuild=False):
        """Build and package a template directory with smart rebuild detection"""
        component_name = directory
        
        # Check if this component needs rebuilding (unless forced)
        if not force_rebuild:
            dependencies = self.get_component_dependencies().get(component_name, [])
            if dependencies:
                
                if component_name == "main":
                    checksum_file = ".checksum"
                elif component_name == "lib":
                    checksum_file = "lib/.checksum"
                else:
                    checksum_file = f"{component_name}/.checksum"
                current_checksum = self.get_component_checksum(*dependencies)
                
                if os.path.exists(checksum_file):
                    with open(checksum_file, "r") as f:
                        stored_checksum = f.read().strip()
                    
                    if current_checksum == stored_checksum:
                        pattern_name = os.path.basename(directory)
                        self.console.print(f"[dim]  {pattern_name}: skipped (no changes detected)[/dim]")
                        return True

        # Use absolute paths to avoid directory changing issues
        abs_directory = os.path.abspath(directory)

        # Track build time
        build_start = time.time()

        try:
            # Build the template from the pattern directory
            cmd = ["sam", "build", "--template-file", "template.yaml"]

            # Add caching but remove parallel flag to avoid race conditions
            # when building multiple templates concurrently
            cmd.extend([
                "--cached",  # Enable SAM build caching
                # Note: Removed --parallel to prevent race conditions with idp_common_pkg
            ])

            if self.use_container_flag and self.use_container_flag.strip():
                cmd.append(self.use_container_flag)

            sam_build_start = time.time()
            self.log_verbose(
                f"Running SAM build command in {directory}: {' '.join(cmd)}"
            )
            # Run SAM build from the pattern directory
            result = subprocess.run(
                cmd, cwd=abs_directory, capture_output=True, text=True
            )
            sam_build_time = time.time() - sam_build_start

            if result.returncode != 0:
                # If cached build fails, try without cache
                self.log_verbose(
                    f"Cached build failed for {directory}, retrying without cache"
                )
                self.console.print(
                    f"[yellow]Cached build failed for {directory}, retrying without cache[/yellow]"
                )
                cmd_no_cache = [c for c in cmd if c != "--cached"]
                sam_build_start = time.time()
                self.log_verbose(
                    f"Running SAM build command (no cache) in {directory}: {' '.join(cmd_no_cache)}"
                )
                result = subprocess.run(
                    cmd_no_cache, cwd=abs_directory, capture_output=True, text=True
                )
                sam_build_time = time.time() - sam_build_start
                if result.returncode != 0:
                    # Log detailed error information
                    error_output = (
                        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                    )
                    self.log_error_details(
                        f"SAM build for {directory}", error_output
                    )
                    return False

            # Package the template (using absolute paths)
            build_template_path = os.path.join(
                abs_directory, ".aws-sam", "build", "template.yaml"
            )
            # Use standard packaged.yaml name to match publish2.sh
            packaged_template_path = os.path.join(
                abs_directory, ".aws-sam", "packaged.yaml"
            )

            cmd = [
                "sam",
                "package",
                "--template-file",
                build_template_path,
                "--output-template-file",
                packaged_template_path,
                "--s3-bucket",
                self.bucket,
                "--s3-prefix",
                self.prefix_and_version,
            ]

            sam_package_start = time.time()
            self.log_verbose(f"Running SAM package command: {' '.join(cmd)}")
            # Run SAM package from project root (no cwd change needed)
            result = subprocess.run(cmd, capture_output=True, text=True)
            sam_package_time = time.time() - sam_package_start

            if result.returncode != 0:
                # Log detailed error information
                error_output = (
                    f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                )
                self.log_error_details(f"SAM package for {directory}", error_output)
                return False

            # Log timing information
            total_time = time.time() - build_start
            pattern_name = os.path.basename(directory)
            self.console.print(
                f"[dim]  {pattern_name}: build={sam_build_time:.1f}s, package={sam_package_time:.1f}s, total={total_time:.1f}s[/dim]"
            )

            # Update component checksum on successful build
            dependencies = self.get_component_dependencies().get(component_name, [])
            if dependencies:
                
                if component_name == "main":
                    checksum_file = ".checksum"
                elif component_name == "lib":
                    checksum_file = "lib/.checksum"
                else:
                    checksum_file = f"{component_name}/.checksum"
                current_checksum = self.get_component_checksum(*dependencies)
                with open(checksum_file, "w") as f:
                    f.write(current_checksum)

        except Exception as e:
            import traceback

            self.log_verbose(f"Exception in build_and_package_template: {e}")
            self.log_verbose(f"Traceback: {traceback.format_exc()}")
            return False

        return True

    def build_patterns_concurrently(self, max_workers=None):
        """Build patterns concurrently with rich progress display"""
        patterns = ["patterns/pattern-1", "patterns/pattern-2", "patterns/pattern-3"]
        existing_patterns = [pattern for pattern in patterns if os.path.exists(pattern)]

        if not existing_patterns:
            self.console.print("[yellow]No patterns found to build[/yellow]")
            return True

        # Check if any patterns have idp_common dependencies
        has_idp_common_deps = self._check_patterns_for_idp_common_deps(
            existing_patterns
        )

        if has_idp_common_deps:
            self.console.print(
                "[yellow]⚠️  idp_common dependencies detected - using sequential builds to prevent race conditions[/yellow]"
            )
            max_workers = 1  # Force sequential builds

        self.console.print(
            f"[cyan]Building {len(existing_patterns)} patterns with {max_workers} workers...[/cyan]"
        )

        # Create progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        ) as progress:
            # Create main task for overall progress
            main_task = progress.add_task(
                "[cyan]Building patterns...", total=len(existing_patterns)
            )

            # Create individual tasks for each pattern
            pattern_tasks = {}
            for pattern in existing_patterns:
                task_id = progress.add_task(
                    f"[yellow]{pattern}[/yellow] - Waiting...", total=1
                )
                pattern_tasks[pattern] = task_id

            # Use ThreadPoolExecutor for I/O bound operations (sam build/package)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Submit all pattern build tasks
                future_to_pattern = {}
                for pattern in existing_patterns:
                    # Update task status to building
                    progress.update(
                        pattern_tasks[pattern],
                        description=f"[yellow]{pattern}[/yellow] - Building...",
                    )
                    future = executor.submit(self.build_and_package_template, pattern)
                    future_to_pattern[future] = pattern

                # Wait for all tasks to complete and check results
                all_successful = True
                completed = 0

                for future in concurrent.futures.as_completed(future_to_pattern):
                    pattern = future_to_pattern[future]
                    completed += 1

                    try:
                        success = future.result()
                        if not success:
                            progress.update(
                                pattern_tasks[pattern],
                                description=f"[red]{pattern}[/red] - Failed!",
                                completed=1,
                            )
                            all_successful = False
                        else:
                            progress.update(
                                pattern_tasks[pattern],
                                description=f"[green]{pattern}[/green] - Complete!",
                                completed=1,
                            )

                        # Update main progress
                        progress.update(main_task, completed=completed)

                    except Exception as e:
                        # Log detailed error information
                        import traceback

                        error_output = f"Exception: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                        self.log_error_details(
                            f"Pattern {pattern} build exception", error_output
                        )

                        progress.update(
                            pattern_tasks[pattern],
                            description=f"[red]{pattern}[/red] - Error: {str(e)[:30]}...",
                            completed=1,
                        )
                        progress.update(main_task, completed=completed)

    def build_patterns_with_smart_detection(self, components_needing_rebuild, max_workers=None):
        """Build patterns with smart dependency detection"""
        patterns = ["patterns/pattern-1", "patterns/pattern-2", "patterns/pattern-3"]
        existing_patterns = [pattern for pattern in patterns if os.path.exists(pattern)]

        if not existing_patterns:
            self.console.print("[yellow]No patterns found to build[/yellow]")
            return True

        # Filter to only patterns that need rebuilding
        patterns_to_rebuild = []
        patterns_to_skip = []
        
        for pattern in existing_patterns:
            needs_rebuild = any(comp['component'] == pattern for comp in components_needing_rebuild)
            if needs_rebuild:
                patterns_to_rebuild.append(pattern)
            else:
                patterns_to_skip.append(pattern)

        # Report what we're doing
        if patterns_to_skip:
            self.console.print(f"[green]✅ Skipping {len(patterns_to_skip)} unchanged patterns: {', '.join([os.path.basename(p) for p in patterns_to_skip])}[/green]")
        
        if not patterns_to_rebuild:
            self.console.print("[green]✅ All patterns are up to date[/green]")
            return True

        # Check if any patterns have lib dependencies (force sequential if so)
        # Check if lib actually changed (not just dependencies exist)
        lib_changed = any(comp['component'] == 'lib' for comp in components_needing_rebuild)
        has_lib_deps = any(
            any('./lib' in dep for dep in comp['dependencies'])
            for comp in components_needing_rebuild
            if comp['component'] in patterns_to_rebuild
        )

        if lib_changed and has_lib_deps:
            self.console.print(
                "[yellow]⚠️  lib dependencies detected - using sequential builds to prevent race conditions[/yellow]"
            )
            max_workers = 1

        self.console.print(
            f"[cyan]Building {len(patterns_to_rebuild)} patterns with {max_workers} workers...[/cyan]"
        )

        # Build using the existing concurrent infrastructure
        return self._build_components_concurrently(patterns_to_rebuild, "patterns", max_workers)

    def build_options_with_smart_detection(self, components_needing_rebuild, max_workers=None):
        """Build options with smart dependency detection"""
        options = ["options/bda-lending-project", "options/bedrockkb"]
        existing_options = [option for option in options if os.path.exists(option)]

        if not existing_options:
            self.console.print("[yellow]No options found to build[/yellow]")
            return True

        # Filter to only options that need rebuilding
        options_to_rebuild = []
        options_to_skip = []
        
        for option in existing_options:
            needs_rebuild = any(comp['component'] == option for comp in components_needing_rebuild)
            if needs_rebuild:
                options_to_rebuild.append(option)
            else:
                options_to_skip.append(option)

        # Report what we're doing
        if options_to_skip:
            self.console.print(f"[green]✅ Skipping {len(options_to_skip)} unchanged options: {', '.join([os.path.basename(p) for p in options_to_skip])}[/green]")
        
        if not options_to_rebuild:
            self.console.print("[green]✅ All options are up to date[/green]")
            return True

        self.console.print(
            f"[cyan]Building {len(options_to_rebuild)} options with {max_workers} workers...[/cyan]"
        )

        # Build using the existing concurrent infrastructure
        return self._build_components_concurrently(options_to_rebuild, "options", max_workers)

    def _build_components_concurrently(self, components, component_type, max_workers):
        """Generic method to build components concurrently with progress display"""
        # Create progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        ) as progress:
            # Create main task for overall progress
            main_task = progress.add_task(
                f"[cyan]Building {component_type}...", total=len(components)
            )

            # Create individual tasks for each component
            component_tasks = {}
            for component in components:
                task_id = progress.add_task(
                    f"[yellow]{component}[/yellow] - Waiting...", total=1
                )
                component_tasks[component] = task_id

            # Use ThreadPoolExecutor for I/O bound operations (sam build/package)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Submit all component build tasks
                future_to_component = {}
                for component in components:
                    # Update task status to building
                    progress.update(
                        component_tasks[component],
                        description=f"[yellow]{component}[/yellow] - Building...",
                    )
                    future = executor.submit(self.build_and_package_template, component, force_rebuild=True)
                    future_to_component[future] = component

                # Wait for all tasks to complete and check results
                all_successful = True
                completed = 0

                for future in concurrent.futures.as_completed(future_to_component):
                    component = future_to_component[future]
                    completed += 1

                    try:
                        success = future.result()
                        if not success:
                            progress.update(
                                component_tasks[component],
                                description=f"[red]{component}[/red] - Failed!",
                                completed=1,
                            )
                            all_successful = False
                        else:
                            progress.update(
                                component_tasks[component],
                                description=f"[green]{component}[/green] - Complete!",
                                completed=1,
                            )

                        # Update main progress
                        progress.update(main_task, completed=completed)

                    except Exception as e:
                        # Log detailed error information
                        import traceback

                        error_output = f"Exception: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                        self.log_error_details(
                            f"{component_type.title()} {component} build exception", error_output
                        )

                        progress.update(
                            component_tasks[component],
                            description=f"[red]{component}[/red] - Error: {str(e)[:30]}...",
                            completed=1,
                        )
                        all_successful = False
                        progress.update(main_task, completed=completed)

        return all_successful

    def _check_patterns_for_idp_common_deps(self, patterns):
        """Check if any patterns have idp_common dependencies in their Lambda functions."""
        for pattern in patterns:
            pattern_path = Path(pattern)
            src_dir = pattern_path / "src"
            if src_dir.exists():
                for func_dir in src_dir.iterdir():
                    if func_dir.is_dir():
                        requirements_file = func_dir / "requirements.txt"
                        if requirements_file.exists():
                            try:
                                content = requirements_file.read_text(encoding="utf-8")
                                if "idp_common" in content:
                                    return True
                            except Exception:
                                continue
        return False

    def build_options_concurrently(self, max_workers=None):
        """Build options concurrently with rich progress display"""
        options = ["options/bda-lending-project", "options/bedrockkb"]
        existing_options = [option for option in options if os.path.exists(option)]

        if not existing_options:
            self.console.print("[yellow]No options found to build[/yellow]")
            return True

        self.console.print(
            f"[cyan]Building {len(existing_options)} options concurrently with {max_workers} workers...[/cyan]"
        )

        # Create progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        ) as progress:
            # Create main task for overall progress
            main_task = progress.add_task(
                "[cyan]Building options...", total=len(existing_options)
            )

            # Create individual tasks for each option
            option_tasks = {}
            for option in existing_options:
                task_id = progress.add_task(
                    f"[yellow]{option}[/yellow] - Waiting...", total=1
                )
                option_tasks[option] = task_id

            # Use ThreadPoolExecutor for I/O bound operations (sam build/package)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Submit all option build tasks
                future_to_option = {}
                for option in existing_options:
                    # Update task status to building
                    progress.update(
                        option_tasks[option],
                        description=f"[yellow]{option}[/yellow] - Building...",
                    )
                    future = executor.submit(self.build_and_package_template, option)
                    future_to_option[future] = option

                # Wait for all tasks to complete and check results
                all_successful = True
                completed = 0

                for future in concurrent.futures.as_completed(future_to_option):
                    option = future_to_option[future]
                    completed += 1

                    try:
                        success = future.result()
                        if not success:
                            progress.update(
                                option_tasks[option],
                                description=f"[red]{option}[/red] - Failed!",
                                completed=1,
                            )
                            all_successful = False
                        else:
                            progress.update(
                                option_tasks[option],
                                description=f"[green]{option}[/green] - Complete!",
                                completed=1,
                            )

                        # Update main progress
                        progress.update(main_task, completed=completed)

                    except Exception as e:
                        # Log detailed error information
                        import traceback

                        error_output = f"Exception: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
                        self.log_error_details(
                            f"Option {option} build exception", error_output
                        )

                        progress.update(
                            option_tasks[option],
                            description=f"[red]{option}[/red] - Error: {str(e)[:30]}...",
                            completed=1,
                        )
                        all_successful = False
                        progress.update(main_task, completed=completed)

        # Store the result for checksum update decision
        self._options_build_successful = all_successful
        return all_successful

    def generate_config_file_list(self):
        """Generate list of configuration files for explicit copying"""
        config_dir = "config_library"
        file_list = []

        for root, dirs, files in os.walk(config_dir):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, config_dir)
                file_list.append(relative_path)

        return sorted(file_list)

    def validate_dependency_configuration(self):
        """Validate that component dependencies are correctly configured."""
        self.console.print("\n[bold cyan]🔍 VALIDATING component dependency configuration[/bold cyan]")
        
        try:
            deps = self.get_component_dependencies()
            main_deps = deps.get('main', [])
            
            # Check if main has lib dependency
            has_lib_dep = './lib' in main_deps
            
            if has_lib_dep:
                self.console.print("[green]✅ Main template correctly includes ./lib dependency[/green]")
                self.log_verbose(f"Main dependencies: {main_deps}")
            else:
                self.console.print("[red]❌ Main template missing ./lib dependency[/red]")
                self.console.print("[red]This will cause main template to not rebuild when lib changes[/red]")
                self.console.print(f"[yellow]Current main dependencies: {main_deps}[/yellow]")
                return False
                
            return True
            
        except Exception as e:
            self.console.print(f"[red]❌ Error validating dependency configuration: {e}[/red]")
            return False

    def validate_lambda_builds(self):
        """Validate that Lambda functions with idp_common_pkg in requirements.txt have the library included."""
        self.console.print(
            "\n[bold cyan]🔍 VALIDATING Lambda builds for idp_common inclusion[/bold cyan]"
        )

        try:
            # Discover functions with idp_common_pkg in requirements.txt
            functions = self._discover_lambda_functions_with_idp_common()

            if not functions:
                self.console.print(
                    "[yellow]No Lambda functions found with idp_common_pkg in requirements.txt[/yellow]"
                )
                return

            self.console.print(
                f"[cyan]📋 Found {len(functions)} Lambda functions with idp_common_pkg in requirements.txt:[/cyan]"
            )
            for func_key, func_info in functions.items():
                self.console.print(f"   • {func_key} → {func_info['function_name']}")
                self.log_verbose(f"     Requirements: {func_info['requirements_msg']}")

            # Validate each function
            all_passed = True
            results = []

            for func_key, func_info in functions.items():
                function_name = func_info["function_name"]
                template_dir = func_info["template_dir"]
                source_path = func_info["source_path"]

                self.log_verbose(f"Validating function: {func_key} → {function_name}")

                # Check if build directory exists and has idp_common
                has_package, issues = self._validate_idp_common_in_build(
                    template_dir, function_name, source_path
                )

                if not has_package:
                    error_msg = f"Missing idp_common: {'; '.join(issues)}"
                    results.append((func_key, False, error_msg))
                    all_passed = False
                    self.log_verbose(f"❌ {func_key}: {error_msg}")
                    continue

                # Test import functionality
                import_success, import_msg = self._test_import_functionality(
                    template_dir, function_name
                )

                if import_success:
                    results.append((func_key, True, "Validation passed"))
                    self.log_verbose(f"✅ {func_key}: All validations passed")
                else:
                    results.append(
                        (func_key, False, f"Import test failed: {import_msg}")
                    )
                    all_passed = False
                    self.log_verbose(f"❌ {func_key}: Import test failed")

            # Print summary
            self.console.print("\n[cyan]📊 Validation Results Summary:[/cyan]")
            self.console.print("=" * 60)

            passed_count = sum(1 for _, passed, _ in results if passed)
            total_count = len(results)

            for func_key, passed, message in results:
                status = "[green]✅ PASS[/green]" if passed else "[red]❌ FAIL[/red]"
                self.console.print(f"{status} {func_key}: {message}")

            self.console.print("=" * 60)
            self.console.print(
                f"Results: {passed_count}/{total_count} functions passed validation"
            )

            if all_passed:
                self.console.print(
                    "[bold green]🎉 All Lambda functions with idp_common_pkg in requirements.txt have the library properly included![/bold green]"
                )
                self.console.print(
                    "[bold green]✅ Lambda build validation passed![/bold green]"
                )
            else:
                self.console.print(
                    "[bold red]💥 Some Lambda functions are missing idp_common library in their builds.[/bold red]"
                )
                self.console.print(
                    "[bold red]❌ Lambda build validation failed![/bold red]"
                )
                self.console.print(
                    "[bold red]🚫 Publish process aborted due to validation failures![/bold red]"
                )
                self.console.print(
                    "[yellow]Fix the missing idp_common dependencies and rebuild before publishing.[/yellow]"
                )
                sys.exit(1)

        except Exception as e:
            self.console.print(
                f"[red]❌ Error running lambda build validation: {e}[/red]"
            )
            if self.verbose:
                import traceback

                self.console.print(f"[red]{traceback.format_exc()}[/red]")
            self.console.print(
                "[bold red]🚫 Publish process aborted due to validation error![/bold red]"
            )
            sys.exit(1)

    def _discover_lambda_functions_with_idp_common(self):
        """Discover all Lambda functions that have idp_common_pkg in requirements.txt."""
        functions = {}
        project_root = Path(__file__).parent.resolve()

        # Check main template Lambda functions
        main_src_dir = project_root / "src" / "lambda"
        if main_src_dir.exists():
            functions.update(
                self._scan_lambda_directory(
                    main_src_dir, project_root / "template.yaml", "main"
                )
            )

        # Check pattern Lambda functions
        patterns_dir = project_root / "patterns"
        if patterns_dir.exists():
            for pattern_dir in patterns_dir.iterdir():
                if pattern_dir.is_dir() and (pattern_dir / "template.yaml").exists():
                    pattern_src = pattern_dir / "src"
                    if pattern_src.exists():
                        functions.update(
                            self._scan_lambda_directory(
                                pattern_src,
                                pattern_dir / "template.yaml",
                                pattern_dir.name,
                            )
                        )

        # Check options Lambda functions
        options_dir = project_root / "options"
        if options_dir.exists():
            for option_dir in options_dir.iterdir():
                if option_dir.is_dir() and (option_dir / "template.yaml").exists():
                    option_src = option_dir / "src"
                    if option_src.exists():
                        functions.update(
                            self._scan_lambda_directory(
                                option_src,
                                option_dir / "template.yaml",
                                option_dir.name,
                            )
                        )

        return functions

    def _scan_lambda_directory(self, src_dir, template_path, context):
        """Scan a directory for Lambda functions that have idp_common_pkg in requirements.txt."""
        functions = {}

        for func_dir in src_dir.iterdir():
            if not func_dir.is_dir():
                continue

            has_idp_common_req, req_msg = self._check_requirements_has_idp_common_pkg(
                func_dir
            )
            if has_idp_common_req:
                function_key = f"{context}/{func_dir.name}"
                functions[function_key] = {
                    "template_path": template_path,
                    "function_name": self._extract_function_name(func_dir.name),
                    "source_path": func_dir,
                    "context": context,
                    "template_dir": template_path.parent,
                    "requirements_msg": req_msg,
                }

        return functions

    def _check_requirements_has_idp_common_pkg(self, func_dir):
        """Check if requirements.txt contains idp_common_pkg dependency."""
        requirements_file = func_dir / "requirements.txt"
        if not requirements_file.exists():
            return False, "No requirements.txt found"

        try:
            content = requirements_file.read_text(encoding="utf-8")
            lines = [
                line.strip()
                for line in content.split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            # Look for idp_common_pkg reference
            for line in lines:
                if "idp_common_pkg" in line or "lib/idp_common_pkg" in line:
                    return True, f"Found dependency: {line}"

            return False, "No idp_common_pkg found in requirements.txt"
        except Exception as e:
            return False, f"Error reading requirements.txt: {e}"

    def _extract_function_name(self, dir_name):
        """Extract CloudFormation function name from directory name."""
        name_mappings = {
            # Pattern functions
            "bda_invoke_function": "InvokeBDAFunction",
            "bda_completion_function": "BDACompletionFunction",
            "processresults_function": "ProcessResultsFunction",
            "summarization_function": "SummarizationFunction",
            "hitl-process-function": "HITLProcessLambdaFunction",
            "hitl-wait-function": "HITLWaitFunction",
            "hitl-status-update-function": "HITLStatusUpdateFunction",
            "ocr_function": "OCRFunction",
            "classification_function": "ClassificationFunction",
            "extraction_function": "ExtractionFunction",
            "assessment_function": "AssessmentFunction",
            # Main template functions
            "queue_processor": "QueueProcessor",
            "workflow_tracker": "WorkflowTracker",
            "evaluation_function": "EvaluationFunction",
            "save_reporting_data": "SaveReportingDataFunction",
            "queue_sender": "QueueSender",
            "analytics_processor": "AnalyticsProcessorFunction",
            "copy_to_baseline_resolver": "CopyToBaselineResolverFunction",
        }
        return name_mappings.get(dir_name, dir_name)

    def _validate_idp_common_in_build(self, template_dir, function_name, source_path):
        """Validate that idp_common package exists in the built Lambda function."""
        build_dir = template_dir / ".aws-sam" / "build" / function_name
        issues = []

        if not build_dir.exists():
            issues.append(f"Build directory not found: {build_dir}")
            return False, issues

        # Check for idp_common directory in build
        idp_common_dir = build_dir / "idp_common"
        if not idp_common_dir.exists():
            issues.append("idp_common directory not found in build")
            return False, issues

        # Check core files
        core_files = ["__init__.py", "models.py"]
        for core_file in core_files:
            file_path = idp_common_dir / core_file
            if not file_path.exists():
                issues.append(f"Missing core file: {core_file}")

        # Check for key modules based on function type
        module_checks = {
            "InvokeBDAFunction": ["bda/bda_service.py", "bda/__init__.py"],
            "BDACompletionFunction": ["metrics/__init__.py"],
            "ProcessResultsFunction": ["ocr/service.py", "extraction/service.py"],
            "ClassificationFunction": ["classification/service.py"],
            "ExtractionFunction": ["extraction/service.py"],
            "OCRFunction": ["ocr/service.py"],
            "AssessmentFunction": ["assessment/service.py"],
            "SummarizationFunction": ["summarization/service.py"],
        }

        if function_name in module_checks:
            for module_path in module_checks[function_name]:
                module_file = idp_common_dir / module_path
                if not module_file.exists():
                    issues.append(f"Missing function-specific module: {module_path}")

        return len(issues) == 0, issues

    def _test_import_functionality(self, template_dir, function_name):
        """Test that idp_common can actually be imported in the built function."""
        build_dir = template_dir / ".aws-sam" / "build" / function_name

        if not build_dir.exists():
            return False, "Build directory not found"

        # Create a test script
        test_script = build_dir / "test_imports.py"
        test_content = """
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    import idp_common
    from idp_common import models
    print("SUCCESS: All imports working")
except ImportError as e:
    print(f"IMPORT_ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
"""

        try:
            test_script.write_text(test_content)

            result = subprocess.run(
                [sys.executable, str(test_script)],
                capture_output=True,
                text=True,
                timeout=30,
            )

            test_script.unlink()  # Clean up

            if result.returncode == 0:
                return True, "Import test passed"
            else:
                return False, f"Import failed: {result.stdout} {result.stderr}"

        except Exception as e:
            if test_script.exists():
                test_script.unlink()
            return False, f"Test execution failed: {e}"

    def upload_config_library(self):
        """Upload configuration library to S3"""
        self.console.print("[bold cyan]UPLOADING config_library to S3[/bold cyan]")
        config_dir = "config_library"

        if not os.path.exists(config_dir):
            self.console.print(
                f"[yellow]Warning: {config_dir} directory not found[/yellow]"
            )
            return

        self.console.print("[cyan]Uploading configuration library to S3[/cyan]")

        # Upload all files in config_library
        for root, dirs, files in os.walk(config_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, config_dir)
                s3_key = f"{self.prefix_and_version}/config_library/{relative_path}"

                try:
                    self.s3_client.upload_file(
                        local_path, self.bucket, s3_key, ExtraArgs={"ACL": self.acl}
                    )
                except ClientError as e:
                    self.console.print(f"[red]Error uploading {local_path}: {e}[/red]")
                    sys.exit(1)

        self.console.print(
            f"[green]Configuration library uploaded to s3://{self.bucket}/{self.prefix_and_version}/config_library[/green]"
        )

    def compute_ui_hash(self):
        """Compute hash of UI folder contents"""
        self.console.print("[cyan]Computing hash of ui folder contents[/cyan]")
        ui_dir = "src/ui"
        return self.get_directory_checksum(ui_dir)

    def package_ui(self):
        """Package UI source code"""
        ui_hash = self.compute_ui_hash()
        zipfile_name = f"src-{ui_hash[:16]}.zip"

        # Check if we need to rebuild
        existing_zipfiles = [
            f
            for f in os.listdir(".aws-sam")
            if f.startswith("src-") and f.endswith(".zip")
        ]

        if existing_zipfiles and existing_zipfiles[0] != zipfile_name:
            self.console.print(
                f"[yellow]WebUI zipfile name changed from {existing_zipfiles[0]} to {zipfile_name}, forcing rebuild[/yellow]"
            )
            # Remove old zipfile
            for old_zip in existing_zipfiles:
                old_path = os.path.join(".aws-sam", old_zip)
                if os.path.exists(old_path):
                    os.remove(old_path)

        zipfile_path = os.path.join(".aws-sam", zipfile_name)

        if not os.path.exists(zipfile_path):
            self.console.print("[bold cyan]PACKAGING src/ui[/bold cyan]")
            self.console.print(f"[cyan]Zipping source to {zipfile_path}[/cyan]")

            with zipfile.ZipFile(zipfile_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                ui_dir = "src/ui"
                for root, dirs, files in os.walk(ui_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, ui_dir)
                        zipf.write(file_path, arcname)

            self.console.print("[cyan]Upload source to S3[/cyan]")
            s3_key = f"{self.prefix_and_version}/{zipfile_name}"

            try:
                self.s3_client.upload_file(
                    zipfile_path, self.bucket, s3_key, ExtraArgs={"ACL": self.acl}
                )
            except ClientError as e:
                self.console.print(f"[red]Error uploading UI zipfile: {e}[/red]")
                sys.exit(1)

        self.console.print(f"[green]WebUI zipfile: {zipfile_name}[/green]")
        return zipfile_name

    def build_main_template(self, webui_zipfile, components_needing_rebuild):
        """Build and package main template with smart detection"""
        self.console.print("[bold cyan]BUILDING main[/bold cyan]")

        # Check if main template needs rebuilding
        main_needs_rebuild = any(comp['component'] == 'main' for comp in components_needing_rebuild)
        
        if main_needs_rebuild or not os.path.exists(".aws-sam/idp-main.yaml"):
            self.console.print("[yellow]Main template needs rebuilding[/yellow]")

            # Build main template
            self.clean_and_build("template.yaml")

            self.console.print("[bold cyan]PACKAGING main[/bold cyan]")

            # Read the template
            with open(".aws-sam/build/template.yaml", "r") as f:
                template_content = f.read()

            # Get configuration file list
            config_files_list = self.generate_config_file_list()
            config_files_json = json.dumps(config_files_list)

            # Get various hashes
            workforce_url_file = "src/lambda/get-workforce-url/index.py"
            a2i_resources_file = "src/lambda/create_a2i_resources/index.py"
            cognito_client_file = "src/lambda/cognito_updater_hitl/index.py"

            workforce_url_hash = (
                self.get_file_checksum(workforce_url_file)[:16]
                if os.path.exists(workforce_url_file)
                else ""
            )
            a2i_resources_hash = (
                self.get_file_checksum(a2i_resources_file)[:16]
                if os.path.exists(a2i_resources_file)
                else ""
            )
            cognito_client_hash = (
                self.get_file_checksum(cognito_client_file)[:16]
                if os.path.exists(cognito_client_file)
                else ""
            )

            # Replace tokens in template
            from datetime import datetime, timezone

            build_date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            replacements = {
                "<VERSION>": self.version,
                "<BUILD_DATE_TIME>": build_date_time,
                "<PUBLIC_SAMPLE_UDOP_MODEL>": self.public_sample_udop_model,
                "<ARTIFACT_BUCKET_TOKEN>": self.bucket,
                "<ARTIFACT_PREFIX_TOKEN>": self.prefix_and_version,
                "<WEBUI_ZIPFILE_TOKEN>": webui_zipfile,
                "<HASH_TOKEN>": self.get_directory_checksum("./lib")[:16],
                "<CONFIG_LIBRARY_HASH_TOKEN>": self.get_directory_checksum(
                    "config_library"
                )[:16],
                "<CONFIG_FILES_LIST_TOKEN>": config_files_json,
                "<WORKFORCE_URL_HASH_TOKEN>": workforce_url_hash,
                "<A2I_RESOURCES_HASH_TOKEN>": a2i_resources_hash,
                "<COGNITO_CLIENT_HASH_TOKEN>": cognito_client_hash,
            }

            self.console.print("[cyan]Inline edit main template to replace:[/cyan]")
            for token, value in replacements.items():
                self.console.print(
                    f"   [yellow]{token}[/yellow] with: [green]{value}[/green]"
                )
                template_content = template_content.replace(token, value)

            # Write the modified template to the build directory
            build_packaged_template_path = ".aws-sam/build/idp-main.yaml"
            with open(build_packaged_template_path, "w") as f:
                f.write(template_content)

            # Package the template from the build directory
            original_cwd = os.getcwd()
            try:
                os.chdir(".aws-sam/build")
                cmd = [
                    "sam",
                    "package",
                    "--template-file",
                    "idp-main.yaml",
                    "--output-template-file",
                    "../../.aws-sam/idp-main.yaml",
                    "--s3-bucket",
                    self.bucket,
                    "--s3-prefix",
                    self.prefix_and_version,
                ]

                self.log_verbose(
                    f"Running main template SAM package command: {' '.join(cmd)}"
                )
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    error_output = (
                        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
                    )
                    self.log_error_details("Main template SAM package", error_output)
                    self.console.print("[red]Error packaging main template[/red]")
                    sys.exit(1)
            finally:
                os.chdir(original_cwd)

            # Update main template checksum on successful build
            dependencies = self.get_component_dependencies().get('main', [])
            if dependencies:
                checksum_file = ".checksum"
                current_checksum = self.get_component_checksum(*dependencies)
                with open(checksum_file, "w") as f:
                    f.write(current_checksum)

        else:
            self.console.print("[green]✅ Main template is up to date[/green]")

        # Always upload the final template to S3, regardless of whether rebuild was needed
        final_template_key = f"{self.prefix}/{self.main_template}"
        packaged_template_path = ".aws-sam/idp-main.yaml"

        # Check if packaged template exists, if not we have a problem
        if not os.path.exists(packaged_template_path):
            self.console.print(
                f"[red]Error: Packaged template not found at {packaged_template_path}[/red]"
            )
            self.console.print(
                "[red]This suggests the template was never built. Try running without cache.[/red]"
            )
            sys.exit(1)

        self.console.print(
            f"[cyan]Uploading main template to S3: {final_template_key}[/cyan]"
        )
        self.log_verbose(f"Local template path: {packaged_template_path}")
        try:
            self.s3_client.upload_file(
                packaged_template_path,
                self.bucket,
                final_template_key,
                ExtraArgs={"ACL": self.acl},
            )
            self.console.print("[green]✅ Main template uploaded successfully[/green]")
        except ClientError as e:
            self.console.print(f"[red]Error uploading main template: {e}[/red]")
            sys.exit(1)

        # Validate the template
        template_url = (
            f"https://s3.{self.region}.amazonaws.com/{self.bucket}/{final_template_key}"
        )
        self.console.print(f"[cyan]Validating template: {template_url}[/cyan]")

        try:
            self.cf_client.validate_template(TemplateURL=template_url)
            self.console.print("[green]✅ Template validation passed[/green]")
        except ClientError as e:
            self.console.print(f"[red]Template validation failed: {e}[/red]")
            sys.exit(1)

    def update_lib_checksum(self):
        """Update lib checksum file to track changes in the library directories"""
        self.console.print(
            "[cyan]Updated lib checksum file to track changes in the library directories[/cyan]"
        )
        lib_checksum = self.get_component_checksum("lib/idp_common_pkg")
        with open("lib/.checksum", "w") as f:
            f.write(lib_checksum)

    def get_source_files_checksum(self, directory):
        """Get checksum of only source code files in a directory"""
        if not os.path.exists(directory):
            return ""

        # Cache directory checksums to avoid recalculation
        cache_key = f"source_checksum_{directory}"
        if hasattr(self, '_checksum_cache') and cache_key in self._checksum_cache:
            return self._checksum_cache[cache_key]
        
        if not hasattr(self, '_checksum_cache'):
            self._checksum_cache = {}

        # Use os.scandir for better performance than os.walk
        checksums = []
        file_count = 0
        
        # Define patterns once
        source_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.yaml', '.yml', '.json', '.txt', '.md', '.toml', '.cfg', '.ini'}
        exclude_dirs = {'__pycache__', '.pytest_cache', '.ruff_cache', 'build', 'dist', '.aws-sam', 'node_modules', '.git', '.vscode', '.idea', 'test-reports', '.coverage', 'htmlcov', 'coverage_html_report', 'tests', 'test'}
        
        def process_directory(dir_path):
            nonlocal file_count
            files_to_process = []
            try:
                with os.scandir(dir_path) as entries:
                    for entry in entries:
                        if entry.is_dir():
                            if entry.name not in exclude_dirs and not entry.name.startswith('.'):
                                process_directory(entry.path)
                        elif entry.is_file():
                            name = entry.name
                            if (not name.startswith('.') and 
                                not name.endswith(('.pyc', '.pyo', '.pyd', '.so', '.log', '.checksum')) and
                                not name.startswith('test_') and not name.endswith('_test.py')):
                                
                                _, ext = os.path.splitext(name)
                                if (ext.lower() in source_extensions or 
                                    name in {'Dockerfile', 'Makefile', 'requirements.txt', 'setup.py', 'setup.cfg'} or
                                    'template' in name.lower()):
                                    files_to_process.append(entry.path)
                
                # Sort files for deterministic order
                for file_path in sorted(files_to_process):
                    relative_path = os.path.relpath(file_path, directory)
                    file_checksum = self.get_file_checksum(file_path)
                    combined = f"{relative_path}:{file_checksum}"
                    checksums.append(hashlib.sha256(combined.encode()).hexdigest())
                    file_count += 1
                    
            except (OSError, PermissionError):
                pass  # Skip inaccessible directories
        
        process_directory(directory)
        
        if self.verbose:
            self.console.print(f"[dim]Checksummed {file_count} source files in {directory}[/dim]")

        # Combine all checksums
        combined = "".join(sorted(checksums))  # Sort for consistency
        result = hashlib.sha256(combined.encode()).hexdigest()
        
        # Cache the result
        self._checksum_cache[cache_key] = result
        return result

    def get_component_checksum(self, *paths):
        """Get combined checksum for component paths (source files only)"""
        # Use instance-level cache to avoid recalculating same paths
        if not hasattr(self, '_component_checksum_cache'):
            self._component_checksum_cache = {}
        
        cache_key = tuple(sorted(paths))
        if cache_key in self._component_checksum_cache:
            return self._component_checksum_cache[cache_key]
        
        checksums = []
        for path in paths:
            if os.path.isfile(path):
                # For individual files, use file checksum
                checksums.append(self.get_file_checksum(path))
            elif os.path.isdir(path):
                # For directories, use source files checksum
                checksums.append(self.get_source_files_checksum(path))

        combined = "".join(checksums)
        result = hashlib.sha256(combined.encode()).hexdigest()
        
        # Cache the result
        self._component_checksum_cache[cache_key] = result
        return result

    def get_component_dependencies(self):
        """Map each component to its dependencies for smart rebuild detection"""
        dependencies = {
            # Main template components
            "main": ["./src", "template.yaml", "./config_library", "./lib"],
            
            # Pattern components  
            "patterns/pattern-1": ["./lib", "patterns/pattern-1/src", "patterns/pattern-1/template.yaml"],
            "patterns/pattern-2": ["./lib", "patterns/pattern-2/src", "patterns/pattern-2/template.yaml"],
            "patterns/pattern-3": ["./lib", "patterns/pattern-3/src", "patterns/pattern-3/template.yaml"],
            
            # Option components (no lib dependency - they don't use idp_common)
            "options/bda-lending-project": ["options/bda-lending-project/src", "options/bda-lending-project/template.yaml"],
            "options/bedrockkb": ["options/bedrockkb/src", "options/bedrockkb/template.yaml"],
            
            # Lib component (tracks changes in idp_common_pkg)
            "lib": ["lib/idp_common_pkg"],
        }
        
        # Filter to only existing components and dependencies
        filtered_dependencies = {}
        for component, deps in dependencies.items():
            if component == "main" or os.path.exists(component):
                existing_deps = [dep for dep in deps if os.path.exists(dep)]
                if existing_deps:
                    filtered_dependencies[component] = existing_deps
        
        return filtered_dependencies

    def get_components_needing_rebuild(self):
        """Determine which components need rebuilding based on dependency changes"""
        dependencies = self.get_component_dependencies()
        components_to_rebuild = []
        
        # Cache checksums to avoid recalculating for shared dependencies (like ./lib)
        checksum_cache = {}
        
        for component, deps in dependencies.items():
            # Use publish2.sh checksum file format: directory/.checksum
            if component == "main":
                checksum_file = ".checksum"
            elif component == "lib":
                checksum_file = "lib/.checksum"
            else:
                checksum_file = f"{component}/.checksum"
            
            # Calculate checksum only once per unique dependency set
            deps_key = tuple(sorted(deps))
            if deps_key not in checksum_cache:
                checksum_cache[deps_key] = self.get_component_checksum(*deps)
            current_checksum = checksum_cache[deps_key]
            
            needs_rebuild = True
            if os.path.exists(checksum_file):
                with open(checksum_file, "r") as f:
                    stored_checksum = f.read().strip()
                needs_rebuild = current_checksum != stored_checksum
            
            if needs_rebuild:
                components_to_rebuild.append({
                    'component': component,
                    'dependencies': deps,
                    'checksum_file': checksum_file,
                    'current_checksum': current_checksum
                })
                
                if self.verbose:
                    self.console.print(f"[yellow]📝 {component} needs rebuild due to changes in: {', '.join(deps)}[/yellow]")
        
        return components_to_rebuild

    def clear_component_cache(self, component):
        """Clear build cache for a specific component"""
        if component == "main":
            sam_dir = ".aws-sam"
        else:
            sam_dir = os.path.join(component, ".aws-sam")
        
        if os.path.exists(sam_dir):
            build_dir = os.path.join(sam_dir, "build")
            cache_dir = os.path.join(sam_dir, "cache")
            
            if os.path.exists(build_dir):
                self.log_verbose(f"Clearing build cache for {component}: {build_dir}")
                shutil.rmtree(build_dir)
            
            if os.path.exists(cache_dir):
                self.log_verbose(f"Clearing SAM cache for {component}: {cache_dir}")
                shutil.rmtree(cache_dir)

    def update_component_checksum(self, component_info):
        """Update checksum for a successfully built component"""
        with open(component_info['checksum_file'], "w") as f:
            f.write(component_info['current_checksum'])
        self.log_verbose(f"Updated checksum for {component_info['component']}")

    def smart_rebuild_detection(self):
        """Perform smart rebuild detection and cache clearing"""
        import os as os_module  # Local import to avoid scoping issues
        
        self.console.print("[cyan]🔍 Analyzing component dependencies for smart rebuilds...[/cyan]")
        
        components_to_rebuild = self.get_components_needing_rebuild()
        
        # Check if lib actually changed (only if checksum exists and differs)
        lib_actually_changed = False
        lib_component = next((comp for comp in components_to_rebuild if comp['component'] == 'lib'), None)
        if lib_component and os_module.path.exists(lib_component['checksum_file']):
            # Lib checksum exists, check if content actually changed
            with open(lib_component['checksum_file'], 'r') as f:
                stored_checksum = f.read().strip()
            current_checksum = lib_component['current_checksum']
            lib_actually_changed = stored_checksum != current_checksum
            
            if self.verbose and lib_actually_changed:
                self.console.print(f"[red]DEBUG: Lib checksum mismatch![/red]")
                self.console.print(f"[red]Stored: {stored_checksum}[/red]")
                self.console.print(f"[red]Current: {current_checksum}[/red]")
        
        if not components_to_rebuild:
            self.console.print("[green]✅ No components need rebuilding[/green]")
            return []
        
        # Filter out lib component ONLY if it's just checksum tracking (not actual changes)
        actual_rebuilds = [comp for comp in components_to_rebuild if not (comp['component'] == 'lib' and not lib_actually_changed)]
        
        if not actual_rebuilds:
            self.console.print("[green]✅ No components need rebuilding[/green]")
            return components_to_rebuild  # Still return lib for checksum update
        
        self.console.print(f"[yellow]📦 {len(actual_rebuilds)} components need rebuilding:[/yellow]")
        
        # Categorize components by what triggered their rebuild
        lib_triggered = []
        component_triggered = []
        config_triggered = []
        lib_component_rebuilding = []
        
        for comp_info in components_to_rebuild:
            component = comp_info['component']
            
            # Handle lib component separately
            if component == 'lib':
                lib_component_rebuilding.append(component)
                continue
            
            # Check what actually changed for this component
            if lib_actually_changed and any('./lib' in dep for dep in comp_info['dependencies']):
                lib_triggered.append(component)
            elif any(component in dep or f"{component}/" in dep for dep in comp_info['dependencies']):
                component_triggered.append(component)
            elif any('config_library' in dep for dep in comp_info['dependencies']):
                config_triggered.append(component)
        
        # Report lib component only if lib source code actually changed
        if lib_component_rebuilding and lib_actually_changed:
            self.console.print(f"   📚 Lib changes detected: {', '.join(lib_component_rebuilding)}")
        
        # Report what triggered rebuilds
        if lib_triggered:
            self.console.print(f"   🔧 Due to lib changes: {', '.join(lib_triggered)}")
            # Only clean lib artifacts when lib source code actually changed (not just missing checksum)
            if lib_actually_changed and os_module.path.exists('lib/.checksum'):
                self.console.print("[yellow]Cleaning lib build artifacts due to library changes[/yellow]")
                self.clean_temp_files(verbose_output=False)
                self.clean_lib(verbose_output=False)
        
        if component_triggered:
            # Separate first build vs actual changes
            changed = []
            first_build = []
            for comp in component_triggered:
                comp_info = next(c for c in components_to_rebuild if c['component'] == comp)
                if os_module.path.exists(comp_info['checksum_file']):
                    changed.append(comp)
                else:
                    first_build.append(comp)
            
            if changed:
                self.console.print(f"   📝 Component changes detected: {', '.join(changed)}")
            if first_build:
                self.console.print(f"   📝 Components (first build): {', '.join(first_build)}")
        
        if config_triggered:
            changed = []
            first_build = []
            for comp in config_triggered:
                comp_info = next(c for c in components_to_rebuild if c['component'] == comp)
                if os_module.path.exists(comp_info['checksum_file']):
                    changed.append(comp)
                else:
                    first_build.append(comp)
            
            if changed:
                self.console.print(f"   ⚙️  Config changes detected: {', '.join(changed)}")
            if first_build:
                self.console.print(f"   ⚙️  Config (first build): {', '.join(first_build)}")
        
        # Clear caches for components that need rebuilding
        for comp_info in components_to_rebuild:
            self.clear_component_cache(comp_info['component'])
        
        # Create lib checksum if missing (lib doesn't go through normal build process)
        lib_component = next((comp for comp in components_to_rebuild if comp['component'] == 'lib'), None)
        if lib_component and not os_module.path.exists(lib_component['checksum_file']):
            # Create lib directory if needed
            os.makedirs(os.path.dirname(lib_component['checksum_file']), exist_ok=True)
            # Write lib checksum
            with open(lib_component['checksum_file'], "w") as f:
                f.write(lib_component['current_checksum'])
            # Remove lib from rebuild list since it's just checksum creation
            components_to_rebuild = [comp for comp in components_to_rebuild if comp['component'] != 'lib']
        
        return components_to_rebuild

    def print_outputs(self):
        """Print final outputs using Rich table formatting"""

        # Generate S3 URL for the main template
        template_url = f"https://s3.{self.region}.amazonaws.com/{self.bucket}/{self.prefix}/{self.main_template}"

        # URL encode the template URL for use in the CloudFormation console URL
        encoded_template_url = quote(template_url, safe=":/?#[]@!$&'()*+,;=")
        launch_url = f"https://{self.region}.console.aws.amazon.com/cloudformation/home?region={self.region}#/stacks/create/review?templateURL={encoded_template_url}&stackName=IDP"

        # Display deployment information first
        self.console.print("\n[bold cyan]Deployment Information:[/bold cyan]")
        self.console.print(f"  • Region: [yellow]{self.region}[/yellow]")
        self.console.print(f"  • Bucket: [yellow]{self.bucket}[/yellow]")
        self.console.print(
            f"  • Template Path: [yellow]{self.prefix}/{self.main_template}[/yellow]"
        )
        self.console.print(
            f"  • Public Access: [yellow]{'Yes' if self.public else 'No'}[/yellow]"
        )

        # SAM deployment commands section
        self.console.print("\n[bold cyan]SAM Deployment Commands:[/bold cyan]")
        
        # Main template commands
        self.console.print(f"\n• [cyan]Main Template:[/cyan]")
        deploy_cmd = f"sam deploy --template-url {template_url} --stack-name IDP --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM --region {self.region}"
        self.console.print(f"  ◦ Deploy: [green]{deploy_cmd}[/green]")
        
        update_cmd = f"sam deploy --template-url {template_url} --stack-name IDP --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM --region {self.region} --no-fail-on-empty-changeset"
        self.console.print(f"  ◦ Update: [green]{update_cmd}[/green]")
        
        validate_cmd = f"aws cloudformation validate-template --template-url {template_url} --region {self.region}"
        self.console.print(f"  ◦ Validate: [green]{validate_cmd}[/green]")

        # Pattern templates
        self.console.print(f"\n• [cyan]Pattern Templates:[/cyan]")
        pattern_templates = ["pattern-1", "pattern-2", "pattern-3"]
        for pattern in pattern_templates:
            pattern_url = f"https://s3.{self.region}.amazonaws.com/{self.bucket}/{self.prefix}/patterns/{pattern}/template.yaml"
            self.console.print(f"  ◦ [yellow]{pattern.title()}:[/yellow]")
            self.console.print(f"    ▪ Deploy: [green]sam deploy --template-url {pattern_url} --stack-name IDP-{pattern.upper()} --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM --region {self.region}[/green]")
            self.console.print(f"    ▪ Validate: [green]aws cloudformation validate-template --template-url {pattern_url} --region {self.region}[/green]")

        # Option templates
        self.console.print(f"\n• [cyan]Option Templates:[/cyan]")
        option_templates = ["bda-lending-project", "bedrockkb"]
        for option in option_templates:
            option_url = f"https://s3.{self.region}.amazonaws.com/{self.bucket}/{self.prefix}/options/{option}/template.yaml"
            self.console.print(f"  ◦ [yellow]{option.replace('-', ' ').title()}:[/yellow]")
            self.console.print(f"    ▪ Deploy: [green]sam deploy --template-url {option_url} --stack-name IDP-{option.upper().replace('-', '_')} --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM --region {self.region}[/green]")
            self.console.print(f"    ▪ Validate: [green]aws cloudformation validate-template --template-url {option_url} --region {self.region}[/green]")

        # Display hyperlinks with complete URLs as the display text
        self.console.print("\n[bold green]Deployment Outputs[/bold green]")

        # 1-Click Launch hyperlink with full URL as display text
        self.console.print("\n[cyan]1-Click Launch (creates new stack):[/cyan]")
        launch_link = f"[link={launch_url}]{launch_url}[/link]"
        self.console.print(f"  {launch_link}")

        # Template URL hyperlink with full URL as display text
        self.console.print("\n[cyan]Template URL (for updating existing stack):[/cyan]")
        template_link = f"[link={template_url}]{template_url}[/link]"
        self.console.print(f"  {template_link}")

    def run(self, args):
        """Main execution method"""
        try:
            # Parse and validate parameters
            self.check_parameters(args)

            # Set up environment
            self.setup_environment()

            # Check prerequisites
            self.check_prerequisites()

            # Ensure .aws-sam directory exists
            self.ensure_aws_sam_directory()

            # Set up S3 bucket
            self.setup_artifacts_bucket()

            # Perform smart rebuild detection and cache management
            components_needing_rebuild = self.smart_rebuild_detection()
            
            # Get components that have lib dependencies and need rebuilding
            lib_dependent_components = [
                comp['component'] for comp in components_needing_rebuild 
                if any('./lib' in dep for dep in comp['dependencies'])
            ]
            
            if lib_dependent_components:
                import os  # Local import to ensure availability
                # Check if lib component actually changed OR if this is first build (lib checksum was just created)
                lib_component_changed = any(
                    comp['component'] == 'lib' and os.path.exists(comp['checksum_file'])
                    for comp in components_needing_rebuild
                )
                
                # Also check if this is effectively a first build (most components missing checksums)
                missing_checksums = sum(1 for comp in components_needing_rebuild 
                                      if not os.path.exists(comp['checksum_file']))
                is_first_build = missing_checksums >= len(components_needing_rebuild) * 0.5  # 50% or more missing
                
                if lib_component_changed or is_first_build:
                    build_type = "Library changes" if lib_component_changed else "First build"
                    self.console.print(
                        f"[yellow]{build_type} detected - will use sequential builds for {len(lib_dependent_components)} components[/yellow]"
                    )
                    self.ensure_idp_common_library_ready()
                else:
                    self.console.print(
                        f"[cyan]Pattern/component changes only - using concurrent builds for {len(lib_dependent_components)} components[/cyan]"
                    )

            # Build patterns and options with smart detection
            self.console.print(
                "[bold cyan]Building components with smart dependency detection...[/bold cyan]"
            )
            start_time = time.time()

            # Determine optimal number of workers
            if self.max_workers is None:
                # Auto-detect: typically CPU count or a bit less, capped at 4
                import os

                self.max_workers = min(4, (os.cpu_count() or 1) + 1)
                self.console.print(
                    f"[green]Auto-detected {self.max_workers} concurrent workers[/green]"
                )

            # Build patterns with smart detection
            self.console.print("\n[bold yellow]📦 Building Patterns[/bold yellow]")
            patterns_start = time.time()
            patterns_success = self.build_patterns_with_smart_detection(
                components_needing_rebuild, max_workers=self.max_workers
            )
            patterns_time = time.time() - patterns_start

            if not patterns_success:
                self.print_error_summary()
                self.console.print(
                    "[red]❌ Error: Failed to build one or more patterns[/red]"
                )
                if not self.verbose:
                    self.console.print(
                        "[dim]Use --verbose flag for detailed error information[/dim]"
                    )
                sys.exit(1)

            # Build options with smart detection
            self.console.print("\n[bold yellow]⚙️  Building Options[/bold yellow]")
            options_start = time.time()
            options_success = self.build_options_with_smart_detection(
                components_needing_rebuild, max_workers=self.max_workers
            )
            options_time = time.time() - options_start

            if not options_success:
                self.print_error_summary()
                self.console.print(
                    "[red]❌ Error: Failed to build one or more options[/red]"
                )
                if not self.verbose:
                    self.console.print(
                        "[dim]Use --verbose flag for detailed error information[/dim]"
                    )
                sys.exit(1)

            total_build_time = time.time() - start_time
            self.console.print(
                f"\n[bold green]✅ Smart build completed in {total_build_time:.2f}s[/bold green]"
            )
            self.console.print(f"   [dim]• Patterns: {patterns_time:.2f}s[/dim]")
            self.console.print(f"   [dim]• Options: {options_time:.2f}s[/dim]")

            # Upload configuration library
            self.upload_config_library()

            # Package UI
            webui_zipfile = self.package_ui()

            # Build main template
            self.build_main_template(webui_zipfile, components_needing_rebuild)

            # Validate dependency configuration
            if not self.validate_dependency_configuration():
                self.console.print("[bold red]🚫 Publish process aborted due to dependency configuration error![/bold red]")
                sys.exit(1)

            # Validate Lambda builds for idp_common inclusion (after all builds complete)
            self.validate_lambda_builds()

            # All builds completed successfully if we reach here
            self.console.print("[green]✅ All builds completed successfully[/green]")

            # Print outputs
            self.print_outputs()

            self.console.print("\n[bold green]✅ Done![/bold green]")

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Operation cancelled by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        publisher = IDPPublisher()
        publisher.print_usage()
        sys.exit(1)

    publisher = IDPPublisher()
    publisher.run(sys.argv[1:])
