Fury - pyBullet Integration Guide
=================================

* Simple Rigid body dynamics

  * Necessary Imports

  * Connection Mode

  * Disconnection

  * Setting Gravity

  * Creating Objects

  * Changing Object Dynamics

  * Adding objects to the scene

  * Application of Force/Torque

  * Enabling Collision

  * Creation of Show Manager

  * Syncing properties of Actors

  * Creation of timer callback

  * Initiating the simulation

  * Rendering multiple objects by a single actor

  * Rendering Joints


* Examples

  * Brick Wall Simulation

  * Ball Collision Simulation

  * Brick Wall Simulation(Single Actor)

  * Chain Simulation

  * Wrecking Ball Simulation


**Official docs:**
  * FURY

  * pyBullet

**NOTE: All elements are in SI units.**


Simple Rigid Body Dynamics
**************************

Necessary Imports
-----------------
The following imports are necessary for physics simulations:

+-----------------------+---------------------------------------------------------------+
|        Imports        |         Usage                                                 |
+=======================+===============================================================+
|         Numpy         |  Creation of arrays and conversion of radians to degrees.     |
+-----------------------+---------------------------------------------------------------+
|         Fury          |  Window and Actor API is used to visualize the simulation.    |
+-----------------------+---------------------------------------------------------------+
|         pyBullet      |  Physics simulation.                                          |
+-----------------------+---------------------------------------------------------------+
|         Itertools     |  The Counter iterator for keeping track of simulation steps.  |
+-----------------------+---------------------------------------------------------------+


.. code-block:: python

  import numpy as np
  from fury import window, actor
  import itertools
  import pybullet as p


Connection Mode
---------------

*“After importing the PyBullet module, the first thing to do is 'connecting' to the physics simulation. PyBullet is designed around a client-server driven API, with a client sending commands and a physics server returning the status. PyBullet has some built-in physics servers: DIRECT and GUI.”*

In our case we use **DIRECT** connection as the visualization will be handled by Fury.

.. code-block:: python

  client = p.connect(p.DIRECT)

*Note: keeping track of physics client ID is optional unless multiple physics clients are used. In order to observe the same simulation in pybullet, replace p.DIRECT with p.GUI.*


Disconnection
-------------

PyBullet Physics client can be shutdown by the following command:

.. code-block:: python

  p.disconnect()

Setting Gravity
---------------

Global Scene gravity can be set using the following command:

.. code-block:: python

  # Gravity vector.
  gravity_x = 0
  gravity_y = 0
  gravity_z = -10
  p.setGravity(gravity_x, gravity_y, gravity_z)

Creating Objects
----------------

The following criterion must be fulfilled in order to create an object which is in sync with both Fury and pyBullet:


+-----------------------+----------------------------------------------------------------------+
|       Object Actor    |         The actor which will be rendered by Fury                     |
+-----------------------+----------------------------------------------------------------------+
|      Collision Shape  |  The shape used by pybullet for collision simulations.               |
|                       |  **Optional** if collision simulation is not required.               |
+-----------------------+----------------------------------------------------------------------+
|       Multi-Body      |  The object that will be tracked by pybullet for general simulations.|
+-----------------------+----------------------------------------------------------------------+

The following is a snippet for creating a spherical ball of radius = 0.3

.. code-block:: python

  ###### Creating BALL
  # Ball actor
  ball_actor = actor.sphere(centers = np.array([[0, 0, 0]]),
                            colors=np.array([1,0,0]),
                            radii=0.3)

  # Collision shape for the ball.
  ball_coll = p.createCollisionShape(p.GEOM_SPHERE,
                                     radius=0.3)

  # Creating a Multibody which will be tracked by pybullet.
  ball = p.createMultiBody(baseMass=3,
                           baseCollisionShapeIndex=ball_coll,
                           basePosition=[2, 0, 1.5],
                           baseOrientation=[ 0, 0, 0, 1 ])

*Note: Centers for the actor must be set to (0, 0, 0) or else the simulation will be offset by that particular value.*


Changing Object Dynamics
------------------------

Object dynamics such as mass, lateral_friction, damping, inertial_pos, inertial_orn, restitution, rolling friction etc can be changed. The following snippet shows how to change the lateral_friction and coeff of restitution of the same ball:

.. code-block:: python

  p.changeDynamics(ball, -1, lateralFriction=0.3, restitution=0.5)

*Note: The second parameter is linkIndex which is for bodies having multiple links or joints. Passing -1 means applying changes to the base object.*

Adding objects to the scene
---------------------------

Objects can be added simply by adding their respective actors to the scene.

.. code-block:: python

  scene = window.Scene()
  scene.add(ball_actor)

Application of Force/Torque
---------------------------

External force or torque to a body can be applied using applyExternalForce and applyExternalTorque. For e.g

.. code-block:: python

  p.applyExternalForce(ball, -1,
                       forceObj=[-2000, 0, 0],
                       posObj=ball_pos,
                       flags=p.WORLD_FRAME)

Here, the first argument refers to the object, the second one refers to the link, forceObj = force vector, posObj = Position Vector of the application of force[Not applicable for applyExternalTorque]. 

.. code-block:: python

  p.applyExternalTorque(ball, -1,
                       forceObj=[-2000, 0, 0],
                       flags=p.WORLD_FRAME)

Enabling collision
------------------
By default, collision detection is enabled between different dynamic moving bodies. The following snippet can be used to enable/disable collision explicitly between a pair of objects.

.. code-block:: python

  enableCol = 1
  p.setCollisionFilterPair(ball, brick, -1, -1, enableCol)

Here, we enable the collision between a ball and a brick object.

