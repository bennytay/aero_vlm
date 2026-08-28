# Data directory

Raw datasets must not be committed. Keep only versioned manifests, licences,
download instructions, hashes, and intentionally tiny test fixtures here.

Phase 1 uses official validation splits and writes raw files under ignored
`data/raw/v0/`. Rebuild the manifest with:

```shell
uv run python -m wam_drones.assemble_v0
```

The source map documents label and licence caveats. In particular, Open Images
does not verify the colour adjectives in the project vocabulary, and
`no_target` is a frame flag rather than an eleventh label.
