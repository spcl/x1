# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions: Afonso Catarino

import argparse

from x1.server.value_server import init_app

parser = argparse.ArgumentParser(description="Training Server")
parser.add_argument(
    "--ipnport", type=int, default=12434, help="Port to run the server on"
)
parser.add_argument("--gpu", type=str, default="cuda", help="GPU to run the server on")
args = parser.parse_args()

app = init_app(args.gpu)

if __name__ == "__main__":
    print(
        f"Starting value server on port {args.ipnport} and putting model on {args.gpu}",
        flush=True,
    )

    app.run(host="0.0.0.0", port=args.ipnport, debug=False)
