=========
Checklist
=========

List of things to update before creating MR for new models:

1. Docs are updated: index.rst as well as addition of any static resources
2. Hydra config is added for the new model
3. Unit test written for the new model as well as any accompanying scripts
4. If you are adding a new dataset class then add it's docs as well
5. All models are configured using yaml, do not set defaults e.g. ``layer=10`` in method definition because they lead to unwarranted assumptions

.. note::

    Currently, we put all documentation related to model in it's source file
    and import it using automodule extension of sphinx. Look at source code
    for existing documentation for examples.

