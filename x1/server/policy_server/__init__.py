# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Eric Schreiber
#
# contributions: Afonso Catarino

from flask import Flask


def init_app(gpu: int) -> Flask:
    """
    Initialize application.

    Args:
        gpu (int): GPU to use.
    Returns:
        Flask: Initialized application.
    """
    app = Flask(__name__)

    with app.app_context():
        from . import config, routes

        app.config.from_object(config.BaseConfig(gpu))

    return app
