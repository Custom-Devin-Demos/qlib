.. _online_serving:

==============
Online Serving
==============
.. currentmodule:: qlib


Introduction
============

.. image:: ../_static/img/online_serving.png
    :align: center


In addition to backtesting, one way to test a model is effective is to make predictions in real market conditions or even do real trading based on those predictions.
``Online Serving`` is a set of modules for online models using the latest data,
which including `Online Manager <#Online Manager>`_, `Online Strategy <#Online Strategy>`_, `Online Tool <#Online Tool>`_, `Updater <#Updater>`_.

`Here <https://github.com/microsoft/qlib/tree/main/examples/online_srv>`_ are several examples for reference, which demonstrate different features of ``Online Serving``.
If you have many models or `task` needs to be managed, please consider `Task Management <../advanced/task_management.html>`_.
The `examples <https://github.com/microsoft/qlib/tree/main/examples/online_srv>`_ are based on some components in `Task Management <../advanced/task_management.html>`_ such as ``TrainerRM`` or ``Collector``.

**NOTE**: User should keep his data source updated to support online serving. For example, Qlib provides `a batch of scripts <https://github.com/microsoft/qlib/blob/main/scripts/data_collector/yahoo/README.md#automatic-update-of-daily-frequency-datafrom-yahoo-finance>`_ to help users update Yahoo daily data.

Known limitations currently
- Currently, the daily updating prediction for the next trading day is supported. But generating orders for the next trading day is not supported due to the `limitations of public data <https://github.com/microsoft/qlib/issues/215#issuecomment-766293563>_`


Online Manager
==============

.. automodule:: qlib.workflow.online.manager
    :members:
    :noindex:

Online Strategy
===============

.. automodule:: qlib.workflow.online.strategy
    :members:
    :noindex:

Online Tool
===========

.. automodule:: qlib.workflow.online.utils
    :members:
    :noindex:

Updater
=======

.. automodule:: qlib.workflow.online.update
    :members:
    :noindex:

Streaming online inference
==========================

``qlib.stream`` layers an in-memory streaming path on top of the batch workflow: a ``StreamSource`` emits ``Tick`` s,
a ``FeatureBuffer`` evaluates the Alpha158/Alpha360 expressions on the buffered window with the regular
``qlib.data.ops`` operators, and an ``OnlineInferenceServer`` scores every completed bar with a trained model and
exposes ``GET /health``, ``GET /signals/latest`` and ``POST /predict``.

``qlib.workflow.online.stream`` connects those live scores to the ``Online Manager`` above without changing it:

- ``StreamSignalSink`` -- a thread-safe ``signal_sink`` callable (``latest()``, ``history(n)``, ``wait(timeout)``).
- ``StreamOnlineStrategy`` -- an ``OnlineStrategy`` that trains nothing (``first_tasks``/``prepare_tasks`` return ``[]``)
  and whose collector yields ``{"pred": DataFrame}`` so ``OnlineManager.prepare_signals()`` / ``get_signals()``
  return the live signals, ready for ``TopkDropoutStrategy(signal=...)``. It can share an ``OnlineManager`` with a
  ``RollingStrategy``.

.. code-block:: python

    from qlib.workflow.online.manager import OnlineManager
    from qlib.workflow.online.stream import StreamOnlineStrategy, StreamSignalSink

    sink = StreamSignalSink()
    server = OnlineInferenceServer.from_recorder(rec_id, experiment_name="exp", source=source, signal_sink=sink)
    om = OnlineManager(StreamOnlineStrategy("live", sink), begin_time="2020-01-01")
    server.start()
    sink.wait(timeout=10)
    om.prepare_signals()
    signals = om.get_signals()   # pd.Series indexed by (datetime, instrument)

The module contract is documented in ``qlib/stream/README.md`` and a runnable walkthrough (synthetic data, offline
training, serving, consuming) lives in ``examples/online_stream/``. The extra dependencies are installed with
``pip install "pyqlib[stream]"``.

.. automodule:: qlib.workflow.online.stream
    :members:
    :noindex:
