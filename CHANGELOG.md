# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.3.0 (2026-04-12)

### Feat

- apply KS4 motion correction when exporting to phy
- populate cluster_group.tsv from KSLabel + decoder_label

### Fix

- remove unsupported overwrite kwarg from Motion.save() in tests (#4)
- stash binary files in parent dir during re-export

## v0.2.0 (2026-03-30)

### Feat

- write binary before spike export, compute PCs from local disk
- make export_phy idempotent for binary write and compression
- add --check-compression flag to skip mtscomp verification
- enable kebab-case CLI flags and subcommand names
- support local paths in discovery functions
- add --compress flag to extract_recording subcommand
- add extract_recording subcommand and rewrite README
- implement S3-to-phy conversion pipeline with CLI

### Fix

- guarantee binary file restoration with try/finally
- stash binary files in parent dir during re-export
- preserve binary files when re-exporting spike data
- enable amplitudes and PC features in phy export
- use thread pool for binary writing instead of fork

### Refactor

- switch CLI from pydantic-settings to tyro
