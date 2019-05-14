#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==================== ------ STD LIBRARIES ------- ====================
import sys, os
import redis
import logging
import argparse
import pathlib
import time, datetime

# ==================== ------ PERSONAL LIBRARIES ------- ====================
sys.path.append(os.path.abspath(os.path.pardir))

from carlhauser_server.Helpers.environment_variable import get_homedir, dir_path
import carlhauser_server.Helpers.json_import_export as json_import_export

import carlhauser_server.Helpers.database_start_stop as database_start_stop
import carlhauser_server.Configuration.database_conf as database_conf

import carlhauser_server.DatabaseAccessor.database_worker as database_accessor

class Database_Requester(database_accessor.Database_Worker):
    # Heritate from the database accesso, and so has already built in access to cache, storage ..

    def __init__(self, conf: database_conf):
        # STD attributes
        super().__init__(conf)

# Launcher for this worker. Launch this file to launch a worker
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch a worker for a specific task : adding picture to database')
    parser.add_argument("-c", '--configuration_file', dest="conf", type=dir_path, help='DB_configuration_file stored as json. Path')
    args = parser.parse_args()

    # Load the provided configuration file and create back the Configuration Object
    conf = database_conf.parse_from_dict(json_import_export.load_json(pathlib.Path(args.conf)))

    # Create the Database Accessor and run it
    db_accessor = Database_Requester(conf)
    db_accessor.run(sleep_in_sec=conf.ADDER_WAIT_SEC)