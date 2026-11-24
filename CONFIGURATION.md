# Private asset configuration

Research storage locations are intentionally absent from this publication.
The public code references logical asset identifiers. Their values must be
supplied by the person running an analysis.

1. Make a private copy of asset_bindings.example.json.
2. Supply the binding values needed by the chosen analysis. Values may identify
   a complete asset, a storage component or a file-selection pattern, as indicated
   by the call site. Keep the profile private and do not commit it.
3. Set WLPR_ASSET_CONFIG to the private configuration selected for the run.
4. Stage the required inputs and verify the applicable source identities and
   content hashes before executing the analysis.

The example contains null values only. There are no built-in research locations,
encoded location strings or automatic fallback profiles. An absent or invalid
binding produces an explicit configuration error without echoing its value.

The public input catalogue associates each input identity with a provenance tier
and a binding identifier. The raw, frozen and derived labels describe provenance;
they do not prescribe a public storage layout.

Bindings are used for input assets, generated outputs and temporary numerical
products. A shared binding must resolve consistently for its producers and
consumers. Directory-like fragments and complete asset references have different
roles; preserve the operation shown at the call site when configuring them.

Use a privately held source manifest to match bindings to research assets.
Neither this publication nor the example reconstructs private storage locations.
Keep scientific parameters, observational periods, units and random seeds
unchanged when adapting the bindings to another machine.
