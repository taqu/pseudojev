// Isolated translation unit: registers sqlite-vec as an auto-extension.
// Including sqlite-vec.h here would pull in sqlite3ext.h which redefines all
// sqlite3_* symbols as macros accessing sqlite3_api, breaking any TU that also
// includes the plain sqlite3.h.  By forward-declaring sqlite3_vec_init we avoid
// that contamination while still registering the extension.

#include "sqlite/sqlite3.h"

struct sqlite3_api_routines;
extern "C" int sqlite3_vec_init(
    sqlite3*                    db,
    char**                      pzErrMsg,
    const sqlite3_api_routines* pApi);

void register_sqlite_vec_autoext()
{
    sqlite3_auto_extension(reinterpret_cast<void (*)(void)>(sqlite3_vec_init));
}
