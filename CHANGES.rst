***************
Release history
***************

.. Changelog entries should follow this format:

   version (release date)
   ======================

   **section**

   - One-line description of change (link to Github issue/PR)

.. Changes should be organized in one of several sections:

   - Added
   - Changed
   - Deprecated
   - Removed
   - Fixed

1.0.0 (1 Dec 2021)
======================
**Description**

First release of the library.

1.0.1 (21 Jan 2022)
======================
**Description**

Name of the library has changed and new models have been added.

- Added: HongTan model
- Changed: The library's name is now TSFEDL.
- Deprecated:
- Removed:
- Fixed: Minor bugs.

1.0.2 (22 Mar 2022)
======================
**Description**

Minor bugs are fixed.

- Added: Documentation of the base model in PyTorch and PyPi configuration files.
- Changed: Changed name of the base module in PyTorch for TSFEDL_BaseModule
- Deprecated:
- Removed:
- Fixed: Minor bugs.

1.0.3 (25 Mar 2022)
======================
**Description**

Minor bugs are fixed, including a valid testing of all the networks.

- Added: Documentation and continuous integration with Travis.
- Changed:
- Deprecated:
- Removed:
- Fixed: Testing scripts.

1.0.5 (31 Mar 2022)
======================
**Description**

Minor bugs are fixed, installation fixed for gpu servers.

- Added: Documentation and gpu testing.
- Changed:
- Deprecated:
- Removed:
- Fixed: Testing scripts and installation.

1.0.6 (5 Jul 2022)
======================
**Description**

Minor bugs are fixed. Unused parameters removed.

- Added:
- Changed: Documentation
- Deprecated:
- Removed: Unused parameters
- Fixed: Testing scripts and installation.

1.0.7 (4 Sep 2022)
======================
**Description**

New models added from 2021 and 2022

- Added: SharPar and DaiXiLi models
- Changed: Documentation
- Deprecated:
- Removed:
- Fixed: Testing scripts and installation.

1.0.7.1 (6 Feb 2023)
======================
**Description**

Installation updated for tensorflow.

- Added:
- Changed:
- Deprecated:
- Removed:
- Fixed: Requirements and config to support only tensorflow as package name.

1.0.7.2 (5 Oct 2023)
======================
**Description**

Installation updated for tensorflow, pytorch and newer versions of Python.

- Added: future compatibility.
- Changed:
- Deprecated:
- Removed:
- Fixed:

1.0.7.4 (31 Oct 2023)
======================
**Description**

Added validation to base module and solved problem to get compatibility with pytorch and pytorch-lightning
latest versions.

- Added: future compatibility, validation step in base module.
- Changed:
- Deprecated:
- Removed:
- Fixed:

1.0.7.5 (8 Nov 2023)
======================
**Description**

Fixed error with ShiHaotian model in Pytorch and top module. Solved and reported in issue #3.

- Added:
- Changed:
- Deprecated:
- Removed:
- Fixed: ShiHaotian model in Pytorch had a default top module behavior

1.0.7.6 (5 Mar 2024)
======================
**Description**

Fixed error with HongTan model, invalid call to arguments.

- Added:
- Changed:
- Deprecated:
- Removed:
- Fixed: HongTan model had an invalid call to arguments