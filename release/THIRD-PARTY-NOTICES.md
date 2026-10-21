# Third-party notices

The pjev executable includes the following third-party software. pjev itself is MIT licensed
(see `LICENSE`).

| Component | Version | License | Copyright |
|---|---|---|---|
| llama.cpp / ggml | 88c4bc60b (PrismML-Eng fork) | MIT | Copyright (c) 2023-2026 The ggml authors |
| cpp-httplib | vendored in llama.cpp | MIT | Copyright (c) 2017 yhirose |
| nlohmann/json | vendored in llama.cpp | MIT | Copyright (c) 2013-2025 Niels Lohmann |
| spdlog | vendored | MIT | Copyright (c) 2016 - present, Gabi Melman and spdlog contributors |
| {fmt} (bundled with spdlog) | vendored | MIT (with optional exception) | Copyright (c) 2012 - present, Victor Zverovich and {fmt} contributors |
| SQLite | 3.53.4 | Public domain | — |
| sqlite-vec | v0.1.9 | MIT or Apache-2.0 (dual, as declared upstream by asg017/sqlite-vec) | Copyright (c) Alex Garcia |
| LZ4 | 1.10.0 | BSD 2-Clause | Copyright (C) 2011-2023, Yann Collet |
| xxHash | bundled with LZ4 | BSD 2-Clause | Copyright (C) 2012-2016, Yann Collet |
| MurmurHash3 | — | Public domain | Austin Appleby |

The models and the exemplar database that pjev loads at run time are **not** part of this package;
they carry their own licenses (inference model: Apache-2.0; nomic-embed-text-v1.5: Apache-2.0 per its
model card; exemplar data: Open-Jev v1.1 — CC0 1.0 / CC BY 4.0 — and OpenJevData-140k — MIT). See the
README section "Runtime resources".

---

## MIT License

Applies to llama.cpp / ggml, cpp-httplib, nlohmann/json, spdlog, {fmt} and sqlite-vec, with the
copyright notices listed above.

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

{fmt} additionally grants: "As an exception, if, as a result of your compiling your source code,
portions of this Software are embedded into a machine-executable object form of such source code, you
may redistribute such embedded portions in such object form without including the above copyright
and permission notices."

## BSD 2-Clause License

Applies to LZ4 and xxHash, with the copyright notices listed above.

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the
distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Public domain

SQLite and MurmurHash3 are in the public domain; no license applies.
