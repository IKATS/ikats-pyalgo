"""
Copyright 2018 CS Systèmes d'Information

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
import logging
import multiprocessing
import time

from ikats.core.library.exception import IkatsConflictError
from ikats.core.resource.api import IkatsApi


class TsCut(object):
    """
    TS Cutting Class
    """

    def __init__(self):
        """
        Cut as TS within a range or with a defined number of points from a start date
        """

        self.logger = logging.getLogger(__name__)

        # Functional identifier
        self.fid = None

        # TSUID
        self._tsuid = None

        # Result containing the data of the new TS
        self.result = None

        # Short name to use to fill in the new functional identifier
        self.short_name = 'cut'

        self.md = None

    @property
    def tsuid(self):
        """
        Getter for self._tsuid
        """
        return self._tsuid

    @tsuid.setter
    def tsuid(self, value):
        """
        Setter for self._tsuid
        """

        self._tsuid = value

        self.fid = IkatsApi.ts.fid(value)

    def cut(self, tsuid=None, sd=None, ed=None, nb_points=None):
        """
        Cut a TS by using the defined options
        :param tsuid: TSUID to cut
        :param sd: the cut starts from this date
        :param ed: the cut ends with this date
        :param nb_points: if defined, the cut will end when the number of points is reached
        :return: the new TS data
        """

        # Check inputs validity
        if tsuid is None or type(tsuid) is not str:
            raise ValueError('valid TSUID must be defined (got %s, type: %s)' % (tsuid, type(tsuid)))
        if ed is None and nb_points is None:
            raise ValueError('A cutting method must be defined between ed and nb_points')
        if ed is not None and nb_points is not None:
            raise ValueError(
                'Only one cutting method must be defined between ed and nb_points (got %s and %s)' % (ed, nb_points))

        self.md = IkatsApi.md.read(ts_list=tsuid)

        if sd is None:
            # Get start date from metadata
            if tsuid not in self.md or 'ikats_start_date' not in self.md[tsuid]:
                raise ValueError("sd must be defined")
            self.logger.info('Start date not provided, using meta data')
            sd = int(self.md[tsuid]['ikats_start_date'])

        # Storing context (and calculate associated functional identifier)
        self.tsuid = tsuid

        # Cutting
        if ed is not None:
            # Cutting from start to end

            if ed <= sd:
                raise ValueError("End date must be after start date")

            self.logger.debug("Cutting using end date %s", ed)
            self.result = IkatsApi.ts.read(tsuid_list=tsuid, sd=sd, ed=ed)[0]

            if len(self.result) != 0:
                if self.result[-1][0] < ed:
                    self.logger.warning("specified end date is greater than the last point date")
        else:
            # Cutting nb_points from start

            try:
                nb_points = int(nb_points)
            except Exception:
                raise TypeError('nb_points must be a valid number')
            try:
                self.result = IkatsApi.ts.read(tsuid_list=tsuid, sd=sd)[0]
            except Exception:
                raise ValueError("No such TS %s" % tsuid)
            if len(self.result) > nb_points:
                self.result = self.result[:nb_points]
            else:
                self.logger.warning(
                    "No cut performed. TS has less points (%s) than requested (%s)", len(self.result), nb_points)

        if len(self.result) == 0:
            self.logger.warning("Cut TS contains no points")

        return self.result

    def save(self, fid=None, tsuid_parent=None):
        """
        Save the cut TS to database

        :param fid: functional Identifier to use
        :param tsuid_parent: the parent tsuid (for inheritance purposes)
        :type fid: str or None
        :type tsuid_parent: str

        :return: the TSUID and the FID of the new TS
        :rtype: dict

        :raise ValueError: if the new TS is empty
        """

        if self.result is None:
            self.logger.warning("Please apply a valid cut action before trying to save")
            return None

        if len(self.result) == 0:
            self.logger.warning("No data to save : please do a cut which produces data")
            return None

        if len(self.result) == 0:
            raise ValueError("The new TS is empty, nothing to store")

        # Use default Functional Identifier if not defined in arguments
        fid = fid or "%s_%s_%d" % (self.fid, self.short_name, time.time() * 1e6)

        # Check if func id already exists in database
        try:
            IkatsApi.fid.tsuid(fid)
            # func id already exist => raise exception
            raise IkatsConflictError("Funcional Id %s already exist in base", fid)
        except ValueError:
            # No match : nominal case
            logging.info("Creating new time series with FID %s in base ", fid)

        # Saving new TS in database
        results = IkatsApi.ts.create(data=self.result, fid=fid, parent=tsuid_parent)

        if 'tsuid' in results:
            result = {
                "tsuid": results['tsuid'],
                "funcId": fid
            }
        else:
            result = None

        return result


def cut_ts(tsuid=None, sd=None, ed=None, nb_points=None, fid=None, save=True):
    """
    Cutting TS wrapper.
    Allow to cut a TS.

    2 methods:
    * Provide a start date and a end date
    * Provide a start date and a number of points to get

    :param tsuid: TSUID to cut
    :param sd: start date
    :param ed: end date
    :param nb_points: number of points
    :param fid: Functional Identifier
    :param save: save flag (True by default)

    :type tsuid: str
    :type sd: int
    :type ed: int
    :type nb_points: int
    :type fid: str or None
    :type save: bool

    :return: the cut TS content (if save=False) or the TSUID + functional identifier (if save=True)

    :raise ValueError: if inputs are not filled properly (see called methods description)
    """
    ts_cut = TsCut()
    ts_cut.cut(tsuid, sd, ed, nb_points)
    if save:
        return ts_cut.save(fid, tsuid)
    else:
        return ts_cut.result


def _cut_ts_multiprocessing(ts, sd, ed, nb_points, save, queue):
    queue.put(cut_ts(tsuid=ts, sd=sd, ed=ed, nb_points=nb_points, fid=None, save=save))


def cut_ds(ds_name=None, sd=None, ed=None, nb_points=None, save=True):
    """
    Cutting dataset wrapper.
    Allow to cut a set of TS (dataset).
    This function uses multiprocessing.

    2 methods:
    * Provide a start date and a end date
    * Provide a start date and a number of points to get

    :param ds_name: name of the dataset to cut
    :param sd: start date
    :param ed: end date
    :param nb_points: number of points
    :param save: save flag (True by default)

    :type ds_name: str
    :type sd: int
    :type ed: int
    :type nb_points: int
    :type save: bool

    :return: the cut dataset content (if save=False) or the (TSUID + functional identifier) list (if save=True)

    :raise ValueError: if inputs are not filled properly (see called methods description)
    """

    # Check inputs validity
    if ds_name is None or type(ds_name) is not str:
        raise ValueError('valid dataset name must be defined (got %s, type: %s)' % (ds_name, type(ds_name)))
    if ed is None and nb_points is None:
        raise ValueError('end date or nb points must be provided to cutting method')
    if ed is not None and nb_points is not None:
        raise ValueError(
            'end date and nb points can not be provided to cutting method together')
    if ed is not None and sd is not None and ed == sd:
        raise ValueError(
            'start date and end date are identical')

    logger = logging.getLogger(__name__)

    jobs = []
    ts_list = IkatsApi.ds.read(ds_name)['ts_list']
    queues = []
    result = []

    for ts in ts_list:
        queue = multiprocessing.Queue()
        queues.append(queue)
        process = multiprocessing.Process(target=_cut_ts_multiprocessing, args=(ts, sd, ed, nb_points, save, queue))
        jobs.append(process)
        process.start()

    for job in jobs:
        logger.debug("Joining cutting job %s", job)
        job.join()

    for queue in queues:
        result.append(queue.get())

    return result
