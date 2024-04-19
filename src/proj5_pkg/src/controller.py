import rospy
import numpy as np
import casadi as ca
from geometry_msgs.msg import Twist

"""
File containing controllers 
Authors: Han Nguyen, Massimiliano de Sa, Spring 2023.
"""

class Controller:
    def __init__(self, observer, trajectory = None, uBounds = None):
        """
        Skeleton class for feedback controllers
        Args:
            observer (Observer): state observer object
            trajectory (Trajectory): trajectory for the controller to track (could just be a constant point!)
            uBounds: minimum and maximum input values to the system
        """
        #store input parameters
        self.observer = observer
        self.trajectory = trajectory
        self.uBounds = uBounds
        self._vmin = uBounds[0][0]
        self._vmax = uBounds[0][1]
        self._wmin = uBounds[1][0]
        self._wmax = uBounds[1][1]

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    
    def eval_input(self, t):
        """
        Solve for and return control input
        Inputs:
            t (float): time in simulation
        Returns:
            u ((Dynamics.inputDimn x 1)): input vector, as determined by controller
        """
        self._u = self.trajectory.vel(t)
        return self._u
    
    def get_input(self):
        """
        Retrieves input stored in class parameter
        Returns:
            self._u: most recent input stored in class paramter
        """
        return self._u
    
    def apply_input(self):
        msg = Twist()
        msg.linear.x = self._u[0]
        msg.angular.z = self._u[1]
        self.pub.publish(msg)
        return

class TurtlebotFBLin:
    def __init__(self, observer, trajectory, frequency, uBounds = None):
        """
        Class for a feedback linearizing controller for a single turtlebot within a larger system
        Args:
            observer (Observer): state observer object
            traj (Trajectory): trajectory object
        """
        #store input parameters
        self.observer = observer
        self.trajectory = trajectory

        #store the time step dt for integration
        self.dt = 1/frequency #from the control frequency in environment.py

        #previous control input
        self.previous_u_input = np.array([0., 0.]).reshape(2,1)

        self.uBounds = uBounds
        self._vmin = uBounds[0][0]
        self._vmax = uBounds[0][1]
        self._wmin = uBounds[1][0]
        self._wmax = uBounds[1][1]

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        #store an initial value for the vDot integral
        self.vDotInt = 0

    def eval_z_input(self, t):
        """
        Solve for the input z to the feedback linearized system.
        Use linear tracking control techniques to accomplish this.
        Inputs:
            t (float): current time
        Returns:
            z ((2x1) NumPy array)
        """
        #get the state of turtlebot i
        qe = self.observer.get_state()

        #get the derivative of q of turtlebot i
        qeDot = self.observer.get_vel(self.previous_u_input)

        #get the trajectory states
        xD, vD, aD = self.trajectory.get_state(t)

        """
        TODO: Your code here:
        Apply feedback linearization to calculate the z input to the system.
        The current state vector, its derivative, and the desired states and their derivatives
        have been extracted above for you. Your code should be the same as the simulation code here.
        """
        #form the augmented state vector.
        qPrime = np.array([qe[0], qe[1], qeDot[0], qeDot[1]])

        #form the augmented desired state vector
        qPrimeDes = np.array([xD[0], xD[1], vD[0], vD[1]])

        #find a control input z to the augmented system
        k1, k2 = 1, 1 #pick k1, k2 s.t. eddot + k2 edot + k1 e = 0 has stable soln
        e = (qPrimeDes - qPrime)[:2]
        eDot = (qPrimeDes - qPrime)[2:]
        z = k1*e + k2*eDot

        #return the z input
        return z
    
    def eval_w_input(self, t, z):
        """
        Solve for the w input to the system
        Inputs:
            t (float): current time in the system
            z ((2x1) NumPy Array): z input to the linear system
        Returns:
            w ((2x1) NumPy Array): w input to the system
        """
        #get the current state
        qe = self.observer.get_state()
        qv = self.observer.get_vel(self.previous_u_input)

        """
        TODO: Your code here
        Apply feedback linearization to calculate the w input to the system.
        The z input to the system has been passed into this function as an argument.
        """

       #get the current phi
        phi = qe[2]

        #get the (xdot, ydot) velocity
        qDot = np.array([qv[0], qv[1]])
        v = np.linalg.norm(qDot)

        #first, eval A(q)
        Aq = np.array([[np.cos(phi), -v*np.sin(phi)],
                       [np.sin(phi),  v*np.cos(phi)]])

        #invert to get the w input - use pseudoinverse to avoid problems
        w = np.linalg.pinv(Aq) @ z

        #return w input
        return w

    def eval_input(self, t):
        """
        Solves for the control input to turtlebot i using a feedback linearizing controller.
        Inputs:
            t (float): current time in simulation
        Returns:
            self._u ((2x1) NumPy array): u input to the turtlebot
        """
        #get the z input to the system
        z = self.eval_z_input(t)

        #get the w input to the system
        w = self.eval_w_input(t, z)

        """
        TODO: Your code here
        Using z and w, calculated from your functions above, calculate u = [v, omega]^T
        to the system. Once you've computed u, store it in self._u
        """

        #integrate the w1 term to get v
        self.vDotInt = self.vDotInt + w[0]*self.dt

        #return the [v, omega] input
        self._u = np.array([[self.vDotInt], [w[1]]])

        #return the [v, omega] input
        return self._u
    
    def get_input(self):
        """
        Retrieves input stored in class parameter
        Returns:
            self._u: most recent input stored in class parameter
        """
        return self._u

    def apply_input(self):
        msg = Twist()
        msg.linear.x = self._u[0][0]
        msg.angular.z = self._u[1][0]
        self.previous_u_input = self._u
        self.pub.publish(msg)
        return

class TurtlebotCBFQP:
    def __init__(self, observer, barriers, trajectory, frequency, uBounds):
        """
        Class for a CBF-QP controller for a single turtlebot within a larger system. This implementation
        applies a CBF-QP directly over the turtlebot feedback linearizing input - it is not nested within the
        feedback linearization step. This CBF-QP should be relative degree 1.

        Args:
            observer (EgoTurtlebotObserver): state observer object for a single turtlebot within the system
            barriers (List of TurtlebotBarrier): List of TurtlebotBarrier objects corresponding to that turtlebot
            traj (Trajectory): trajectory object
        """
        #store input parameters
        self.observer = observer
        self.barriers = barriers
        self.trajectory = trajectory

        #create a nominal controller
        self.nominalController = TurtlebotFBLin(observer, trajectory,frequency, uBounds)

        #store the input parameter (should not be called directly but with get_input)
        self._u = np.zeros((2, 1))

        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    def eval_input(self, t):
        """
        Solves for the control input to turtlebot i using a CBF-QP controller.
        Inputs:
            t (float): current time in simulation
        Returns:
            self._u ((2x1) NumPy array): safe input vector
        """
        #get the state vector of turtlebot i
        q = self.observer.get_state()

        #get the nominal control input to the system using the controller written above
        kX = self.nominalController.eval_input(t)

        """
        TODO: Your code here
        Apply a CBF-QP directly around the nominal control input kx = [v, omega]^T from the 
        tracking controller. As opposed to the CBF-QP implemented in simulation, this CBF-QP
        should only involve the first derivative of h in the optimization constraint.
        
        The barrier function objects are stored in self.barriers, which is a list containing 
        the barrier function objects for the system. To evaluate the value of a barrier function, you
        can use:
            h, hDot = bar.eval(u, t)
        Where bar is one of the barrier function objects stored in the self.barriers list and u is the 
        input vector from casadi.
        """

        #TODO: Apply casadi to solve for the safe input u

        #set up the optimization problem
        opti = ca.Opti()

        #define the decision variable
        u = opti.variable(2)

        #define gamma for CBF tuning
        gamma = 1

        #apply the N-1 barrier constraints
        for bar in self.barriers:
            #get the values of h and hDot
            h, hDot = bar.eval(u, t)

            #compute the optimization constraint
            opti.subject_to(hDot >= -gamma*h)

        cost = ca.norm_2(u - kX)**2

        opti.minimize(cost)
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)

        #solve optimization
        try:
            sol = opti.solve()
            uOpt = sol.value(u) #extract optimal input
            solverFailed = False
        except:
            print("Solver failed!")
            solverFailed = True
            uOpt = np.zeros((2, 1))
        #evaluate and return the input
        self._u = uOpt.reshape((2, 1))

        #return the safe input
        return self._u
    
    def get_input(self):
        """
        Retrieves input stored in class parameter
        Returns:
            self._u: most recent input stored in class parameter
        """
        return self._u

    def apply_input(self):
        msg = Twist()
        msg.linear.x = self._u[0][0]
        msg.angular.z = self._u[1][0]
        self.previous_u_input = self._u
        self.pub.publish(msg)
        return