#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==================== ------ STD LIBRARIES ------- ====================
import sys, os
import redis
import logging
import time
import argparse
import pathlib
# ==================== ------ PERSONAL LIBRARIES ------- ====================
sys.path.append(os.path.abspath(os.path.pardir))

from carlhauser_server.Helpers.environment_variable import get_homedir, dir_path
import carlhauser_server.Helpers.json_import_export as json_import_export

import carlhauser_server.DatabaseAccessor.database_worker as database_accessor
import carlhauser_server.Configuration.database_conf as database_conf
import carlhauser_server.Configuration.feature_extractor_conf as feature_extractor_conf

import carlhauser_server.FeatureExtractor.picture_hasher as picture_hasher
import carlhauser_server.FeatureExtractor.picture_orber as picture_orber
import carlhauser_server.Helpers.database_start_stop as database_start_stop


class Feature_Worker(database_accessor.Database_Worker):
    # Heritate from the database accesso, and so has already built in access to cache, storage ..

    def __init__(self, db_conf: database_conf, fe_conf: feature_extractor_conf):
        # STD attributes
        super().__init__(db_conf)
        self.fe_conf = fe_conf
        self.picture_hasher = picture_hasher.Picture_Hasher(fe_conf)
        self.picture_orber = picture_orber.Picture_Orber(fe_conf)

        # Get sockets
        tmp_db_handler = database_start_stop.Database_StartStop(conf=self.conf)

        # reconnect to storages, without decoding
        self.storage_db = redis.Redis(unix_socket_path=tmp_db_handler.get_socket_path('storage'), decode_responses=False)
        self.cache_db = redis.Redis(unix_socket_path=tmp_db_handler.get_socket_path('cache'), decode_responses=False)


    def _to_run_forever(self):
        self.process_picture()

    def process_picture(self):
        # Method called infinitely, in loop

        # Trying to fetch from queue
        fetched_id, fetched_dict = self.get_from_queue(self.cache_db, self.input_queue)

        # If there is nothing fetched
        if not fetched_id:
            # Nothing to do
            time.sleep(0.1)
            return 0

        try:
            self.logger.info(f"Feature worker processing {fetched_id}")

            # Get picture from picture_id
            picture = fetched_dict[b"img"]
            self.logger.info(f"Loaded picture {type(picture)}")

            # Get hash values of picture
            hash_dict = self.picture_hasher.hash_picture(picture)
            self.logger.debug(f"Computed hashes : {hash_dict}")

            # Get ORB values of picture
            orb_dict = self.picture_orber.orb_picture(picture)
            self.logger.debug(f"Computed orb values : {orb_dict}")

            # Merge dictionaries
            merged_dict = {**hash_dict, **orb_dict}
            self.logger.debug(f"To send to db dict : {merged_dict}")

            # Remove old data and send dictionary in hashmap to redis
            # TODO : self.cache_db.del(fetched_id)
            self.add_to_queue(self.cache_db, self.ouput_queue, fetched_id, merged_dict, pickle=True)

        except:
            return 1


# Launcher for this worker. Launch this file to launch a worker
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch a worker for a specific task.')
    parser.add_argument("-c", '--configuration_file', dest="conf_db", type=dir_path, help='DB_configuration_file stored as json. Path')
    parser.add_argument("-cfe", '--feature_conf_file', dest="conf_fe", type=dir_path, help='Feature_Extrator_configuration stored as json. Path')
    parser.add_argument("-m", '--mode', dest="mode", required=True, type=str, choices={"ADD", "REQUEST"}, help='Specify queues to work from/to for the worker.')
    # parser.add_argument("-iq", '--input_queue', dest="input_queue", type=str, help='Specify input queue')
    # parser.add_argument("-oq", '--output_queue', dest="output_queue", type=str, help='Specify output queue')

    args = parser.parse_args()

    # Load the provided configuration file and create back the configuration object
    db_conf = database_conf.parse_from_dict(json_import_export.load_json(pathlib.Path(args.conf_db)))
    fe_conf = feature_extractor_conf.parse_from_dict(json_import_export.load_json(pathlib.Path(args.conf_fe)))

    # Create the Feature Worker and run it
    feature_worker = Feature_Worker(db_conf=db_conf, fe_conf=fe_conf)
    if args.mode == "ADD":
        feature_worker.input_queue = "feature_to_add"
        feature_worker.ouput_queue = "db_to_add"
        feature_worker.run(sleep_in_sec=fe_conf.FEATURE_ADDER_WAIT_SEC)
    elif args.mode == "REQUEST":
        feature_worker.input_queue = "feature_to_request"
        feature_worker.ouput_queue = "db_to_request"
        feature_worker.run(sleep_in_sec=fe_conf.FEATURE_REQUEST_WAIT_SEC)
    else:
        print("ARG_PARSER didn't do his job : you should provide a mode for the worker, to know what to do : from where to get pictures to hash, and here to where to put the result back")
