# Runtime upgrade

- [ ] Import and verify the published WMMA runtime image.
- [x] Verify and commit the deployment profile to fork main (`de5fc7a`, pushed).
- [ ] Build and activate the persistent Nix service configuration.
- [ ] Restart and validate inference, memory and speed; roll back on failure.
- [ ] Record final live image and verification results.

## Current blockers

- Published image downloads repeatedly stall/time out; partial downloads and
  resume metadata are retained. Do not switch before complete hash verification.
- Full NixOS build fails in unrelated `open-bambu-networking`.
- The r9v unit builds independently at
  `/nix/store/r73n90d749vydq7avk71waiwng00qch6-unit-r9v.service`.
- Nix configuration commit `de1ae9f` exists locally and in `/etc/nixos`.
  Its private Git push was rejected for remote changes, then fetch failed
  because the server refused the SSH connection. Reconcile before pushing.
- Production still runs rollback image `580bbcecccb3`; no activation/restart
  has occurred in this upgrade attempt. Both services remain active.
- Existing `/etc/systemd/system.control/r9v.service.d/90-accepted-context.conf`
  overrides the image. Preserve a copy and reconcile that override at cutover;
  changing the Nix unit alone will not select the new image.
