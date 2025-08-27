# Importing Packages
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000 # Fixes OverflowError: Exceeded cell block limit

### Changeable variables ###
# These variable are designed to be changed giving longer time frame, change the equations, mass, etc. Go crazy with the values. 

# Gravitational constant 

G = 4 * (np.pi ** 2)    #  4 * (np.pi ** 2) AU^3/M_sun yr^2

# Masses of Sun, Earth and Jupiter in Solar masses

Ms = 1 # Mass of Sun in solar masses, by default it is 1
    
Me = 1/(332950) # Mass of Earth in solar masses, by default it is 1/(332950)

Mj = 1/(1048) # Mass of Jupiter in solar masses, by default it is 1/(1048)

# Equation variables

N = 1 # This variable is on the numerator of all the equations, by default it is 1

D = 1 # This variable is on the denominator of all the equations, by default it is 1

P = 3/2 # This it the power the denominator will be taken to, by default it is 3/2

# Length of simulation in years

max_t = 10000 # Anything above 100000 we recommend getting a coffee or some food while waiting, also adjust the step size. By default it is 10000

# Intial velocity of planets in AU/Yr 

Earth_vx = 0.0 # Earth's initial velocity in x-direction, we use 0.0 for our data

Earth_vy = 6.1795 # Earth's initial velocity in y-direction, we used 6.179

Jupiter_vx = -2.8921 # Jupiter's intial velocity in x-direction, we used -2.8921 for our data

Jupiter_vy = 0 # Jupiter's intial velocity in y-direction, we used 0 for our data

# Speed of animation

s = 10


### Define Models ###

# EJS Model

def EJSmodel(state, t):
    """
    Inputs: 
    
    state array: This contains information of Earth's initial position and velocity and Jupiter's initial position and velocity in the following order: [Earth x-positon, Earth y-position, Earth x-velocity, Earth y-velocity, Jupiter x-positon, Jupiter y-position, Jupiter x-velocity, Jupiter y-velocity]
    
    t: Time array, contians information about how long the simulation will run anf affects the step size of odeint.

    Returns:

    EJdstate_dt: The derivatives of Earth and Jupiter's position and velocity. The array is in the same order as state array. [Derivative of Earth x-position, ..., Derivative of Jupiter y-velocity]
    
    """

    # Earth variables

    Ex = state[0]
    Ey = state[1]
    Ex_dot = state[2]
    Ey_dot = state[3]

    # Jupiter variables

    Jx = state[4]
    Jy = state[5]
    Jx_dot = state[6]
    Jy_dot = state[7]

    # Equations

    Ex_ddot = -G * Ms * ((Ex * N) / (D * (Ex ** 2 + Ey ** 2)) ** (P)) -G * (Mj) * (((Ex - Jx) * N) / (D * ((Ex - Jx) ** 2 + (Ey - Jy) ** 2)) ** (P))

    Ey_ddot = -G * Ms * ((Ey * N) / (D * (Ex ** 2 + Ey ** 2)) ** (P)) -G * (Mj) * (((Ey - Jy) * N) / (D * ((Ex - Jx) ** 2 + (Ey - Jy) ** 2)) ** (P))

    Jx_ddot = -G * Ms * ((Jx * N) / (D * (Jx ** 2 + Jy ** 2)) ** (P)) +G * (Me) * (((Ex - Jx) * N) / (D * ((Ex - Jx) ** 2 + (Ey - Jy) ** 2)) ** (P))

    Jy_ddot = -G * Ms * ((Jy * N) / (D * (Jx ** 2 + Jy ** 2)) ** (P)) +G * (Me) * (((Ey - Jy) * N) / (D * ((Ex - Jx) ** 2 + (Ey - Jy) ** 2)) ** (P))

    EJdstate_dt = [Ex_dot, Ey_dot, Ex_ddot, Ey_ddot, Jx_dot, Jy_dot, Jx_ddot, Jy_ddot]

    return EJdstate_dt

# ES Model

def ESmodel(state, t):
    """
    Inputs: 
    
    state array: This contains information of Earth's initial position and velocity in the following order: [Earth x-positon, Earth y-position, Earth x-velocity, Earth y-velocity]
    
    t: Time array, contians information about how long the simulation will run anf affects the step size of odeint.

    Returns:

    Edstate_dt: The derivatives of Earth's position and velocity. The array is in the same order as state array. [Derivative of Earth x-position, ..., Derivative of Earth y-velocity]
    
    """
    
    # Define constants
    
    Ex = state[0]
    
    Ey = state[1]
    
    Ex_dot = state[2]
    
    Ey_dot = state[3]
    
    # Equations

    Ex_ddot = -G * Ms *  (N * Ex) / (D * (Ex ** 2 + Ey ** 2)) ** (P)
   
    Ey_ddot = -G * Ms *  (N * Ey) / (D * (Ex ** 2 + Ey ** 2)) ** (P)
  
    Edstate_dt = [Ex_dot, Ey_dot, Ex_ddot, Ey_ddot]
    
    return Edstate_dt

# JS Model

def JSmodel(state, t):
    """
    Inputs: 
    
    state array: This contains information of Jupiter's initial position and velocity in the following order: [Jupiter x-positon, Jupiter y-position, Jupiter x-velocity, Jupiter y-velocity]
    
    t: Time array, contians information about how long the simulation will run and affects the step size of odeint.

    Returns:

    Jdstate_dt: The derivatives of Earth's position and velocity. The array is in the same order as state array. [Derivative of Jupiter x-position, ..., Derivative of Jupiter y-velocity]
    
    """

    #Define constants

    Jx = state[0]

    Jy = state[1]

    Jx_dot = state[2]

    Jy_dot = state[3]

    # Equations

    Jx_ddot = -G * Ms * (N * Jx) / (D * (Jx ** 2 + Jy ** 2) ** (P))

    Jy_ddot = -G * Ms * (N * Jy) / (D * (Jx ** 2 + Jy ** 2) ** (P))

    Jdstate_dt = [Jx_dot, Jy_dot, Jx_ddot, Jy_ddot]

    return Jdstate_dt


### Earth Initial conditions ###

# Position

EX_0 =  1.01671123 # [AU] Aphelion 1.01671123 = 1(1 + 0.01671123)
EY_0 = 0  # [AU]

# Velocity

EVX_0 = float(Earth_vx)  # [AU/Yr]
EVY_0 = float(Earth_vy)  # [AU/Yr]

### Jupiter Inital conditions ###

# Position

JX_0 = 0 # [AU]
JY_0 = 4.950429 # [AU] Perihelion

# Velocity

JVX_0 = float(Jupiter_vx) # [AU/Yr] 
JVY_0 = float(Jupiter_vy)  # [AU/Yr]  

### Place intial conditions in a vector ###

EJstate_0 = [EX_0, EY_0, EVX_0, EVY_0, JX_0, JY_0, JVX_0, JVY_0]

### Time Array ###

t = np.linspace(0, max_t, 1000000)  # Simulates for a time period of t_max years. 

### Solving ODE ###

# ODE solver function

EJsol = odeint(EJSmodel, EJstate_0, t)

ESsol = odeint(ESmodel, EJstate_0[0:4], t)

JSsol = odeint(JSmodel, EJstate_0[4:], t)

# EJS-Earth Solutions

X_EJS_Ear = EJsol[:, 0]  # X-coord [AU] of Earth over time interval 
Y_EJS_Ear = EJsol[:, 1]  # Y-coord [AU] of Earth over time interval
VX_EJS_Ear = EJsol[:, 2] # X-Velocity [AU/yr] of Earth over time interval
VY_EJS_Ear = EJsol[:, 3] # Y-Velocity [AU/yr] of Earth over time interval

# EJS-Jupiter Solutions

X_EJS_Jup = EJsol[:, 4]  # X-coord [AU] of Jupiter over time interval 
Y_EJS_Jup = EJsol[:, 5]  # Y-coord [AU] of Jupiter over time interval
VX_EJS_Jup = EJsol[:, 6] # X-Velocity [AU] of Jupiter over time interval
VY_EJS_Jup = EJsol[:, 7] # Y-Velocity [AU] of Jupiter over time interval

# ES-Earth Solutions

X_ES_Ear = ESsol[:, 0]  # X-coord [AU] of Earth over time interval 
Y_ES_Ear = ESsol[:, 1]  # Y-coord [AU] of Earth over time interval
VX_ES_Ear = ESsol[:, 2] # X-Velocity [AU/yr] of Earth over time interval
VY_ES_Ear = ESsol[:, 3] # Y-Velocity [AU/yr] of Earth over time interval

# JS-Jupiter Solutions

X_JS_Jup = JSsol[:, 0]  # X-coord [AU] of Jupiter over time interval 
Y_JS_Jup = JSsol[:, 1]  # Y-coord [AU] of Jupiter over time interval
VX_JS_Jup = JSsol[:, 2] # X-Velocity [AU/yr] of Jupiter over time interval
VY_JS_Jup = JSsol[:, 3] # Y-Velocity [AU/yr] of Jupiter over time interval

### Plotting ES Model ###

plt.figure()
plt.plot(X_ES_Ear, Y_ES_Ear, 'blue')
plt.plot(0, 0,'yo', label = 'Sun')      # Yellow marker for Sun's position
plt.plot(X_ES_Ear[0], 0, 'bo', label = 'Earth Initial position') # Blue marker for Earths original position

plt.axis('equal')
plt.xlabel ('X [AU]')
plt.ylabel ('Y [AU]')
plt.title("Earth\'s orbit after {} years".format(max_t))
plt.legend(loc = 'lower right')
plt.savefig("ES_model")
plt.show()

### Plotting JS Model ###

plt.figure()
plt.plot(X_JS_Jup, Y_JS_Jup, 'red')
plt.plot(0, 0,'yo', label = 'Sun')      # Yellow marker for Sun's position
plt.plot(0, Y_JS_Jup[0], 'ro', label = 'Jupiter Initial position')

plt.axis('equal')
plt.xlabel ('X [AU]')
plt.ylabel ('Y [AU]')
plt.title("Jupiter\'s orbit after {} years".format(max_t))
plt.legend(loc = 'lower right')
plt.savefig("JS_model")
plt.show()

### Plotting EJS model ###

plt.figure()

# Sun

plt.plot(0, 0,'yo', label = 'Sun')  # Yellow marker for Sun's position

# Earth

plt.plot(X_EJS_Ear, Y_EJS_Ear, 'blue', label = 'Earth\'s Orbit')

plt.plot(X_EJS_Ear[0], 0, 'bo', label = 'Earth Initial position') # Blue marker for Earth's original position

# Jupiter

plt.plot(X_JS_Jup, Y_JS_Jup, 'red', label = 'Jupiter\'s Orbit')

plt.plot(0, Y_JS_Jup[0], 'ro', label = 'Jupiter Initial position') # Red marker for Jupiter's original position

# Plot

plt.axis('equal')

plt.xlabel ('X [AU]')

plt.ylabel ('Y [AU]')

plt.title("Earth and Jupiter\'s orbit after {} years".format(max_t))

plt.legend(loc = 'lower right')
plt.savefig("EJS_model")
plt.show()

### Animation function ###
# To make the animation work in python go to Tools > Preferences > IPython Console > Graphics > Backend and change it from "Inline" to "Automatic"

## EJS Animation ##

# Setting up data for animation

trail = 1 # Trail behind the planet

fig, ax = plt.subplots()

# Define animation function

def EJS_animator(i, trail=1):

    Jupiter_EJS.set_data(X_EJS_Jup[(i - trail) * s:i * s], Y_EJS_Jup[(i - trail) * s :i * s])

    Earth_EJS.set_data(X_EJS_Ear[(i - trail) * s:i * s], Y_EJS_Ear[(i - trail) * s:i * s]) # i * s skips frames to give faster animation

    ax.set(xlim=(-6.5, 6.5), ylim=(-6.5, 6.5), xlabel='X [AU]', ylabel='Y [AU]') # Ensure animation looks good, if using G = 4pi**2, change limit: xlim=(-15, 15), ylim=(-20, 6.5)
    
    ax.set_title("Earth and Jupiter\'s orbit \n Time: {} yrs".format(np.round(t[i * s], decimals = 2)))

    return Earth_EJS, Jupiter_EJS,


# Call the function

Earth_EJS, = ax.plot([], [], 'o-', markevery= [-1], label = "Earth")

Jupiter_EJS, = ax.plot([], [], 'ro-', markevery= [-1], label = "Jupiter")

Sun_EJS, = ax.plot([0], [0], 'yo', label = "Sun")

ani = animation.FuncAnimation(fig, EJS_animator, frames=len(t), fargs=(trail,), interval=100, blit=False)

plt.legend()
plt.show()

## Save animation ##
# Error running in spyder, affects saving animation

#saveani  = animation.FuncAnimation(fig, EJS_animator, frames=200, fargs=(trail,), interval=10, blit=False) # New animation fucntion with less frames so we can save the gif.

#saveani.save("EJSmodel-T_{}-S_{}-ANI.gif".format(max_t, s), dpi=300, fps=30, bitrate=-1)