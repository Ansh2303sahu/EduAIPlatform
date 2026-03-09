# Database Backup & Restore Procedure

## Automatic Backups
- Supabase automated daily backups enabled
- Retention managed by Supabase plan
- Backups include Postgres data (tables, RLS, policies)

## Restore Procedure (Dashboard)
1. Open Supabase Dashboard
2. Go to Project → Settings → Database → Backups
3. Select a backup timestamp
4. Restore to:
   - Same project (overwrite), or
   - New temporary project (recommended for verification)
5. Verify restored data:
   - Tables exist
   - RLS policies intact
   - Storage metadata intact

## Emergency Notes
- Storage objects remain in bucket storage
- Metadata (`files` table) restored with DB
- Signed URL access resumes normally
