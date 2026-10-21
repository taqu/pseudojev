#pragma once

// Register sqlite_vec as a SQLite auto-extension.
// Must be called once before any sqlite3_open() that needs vector support.
// Defined in its own translation unit to avoid sqlite3ext.h macro contamination.
void register_sqlite_vec_autoext();
