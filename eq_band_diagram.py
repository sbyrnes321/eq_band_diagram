# -*- coding: utf-8 -*-
"""
Written by Steve Byrnes, 2012 -- http://sjbyrnes.com - steven.byrnes@gmail.com

Calculates equilibrium band structures of 1D semiconductor stacks. (i.e., it
solves the poisson-boltzmann equation by the finite differences method.)

This is the only module in the package. It's written in Python 2.7.

See http://packages.python.org/eq_band_diagram for general discussion and
overview. The functions are all described in their docstrings (below). There
is no other documentation besides that.

Try running example1(), example2(), ..., example5(). If you look at the code
for those, it can be a starting-point for your own calculations.
"""
#Copyright (C) 2012 Steven Byrnes
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import division, print_function

import numpy as np
import math
import matplotlib.pyplot as plt
inf = float('inf')

# Boltzmann constant times the semiconductor temperature, expressed in eV
# I'm assuming 300 K.
kT_in_eV = 0.02585

# e > 0 is the elementary charge. We are expressing charge densities in
# e/cm^3, voltages in volts, and distances in nanometers. So the value of
# epsilon_0 (permittivity of free space) is expressed in the strange units
# of ((e/cm^3) / (V/nm^2))
eps0_in_e_per_cm3_over_V_per_nm2 = 5.5263e19


############################################################################
############################# CORE CALCULATION #############################
############################################################################


def local_charge(Evac_minus_Ei, ni, charge_from_dopants, Evac_minus_EF):
    """
    Calculate local charge. This function is VECTORIZED, i.e. if all the
    inputs are numpy arrays of the same size, so is the output. (But the
    function also works for float inputs.) (!!!!! CHECK !!!!)
    
    Inputs
    ------
    
    * Evac_minus_Ei is the [positive] difference (in eV) between the local
      vacuum level and the intrinsic fermi level (level at which p=n).
      
    * ni = intrinsic electron concentration in cm^-3 (defined by p = n = ni
      when the undoped material is in thermal equilibrium)
    
    * charge_from_dopants (in e/cm^3) equals the density of ionized donors
      minus the density of ionized acceptors. (e>0 is the elementary charge.)
    
    * Evac_minus_EF is the [positive] difference (in eV) between the local
      vacuum level and fermi level.
    
    Output
    ------
    
    Outputs a dictionary with entries:
    
    * n, the density of free electrons in cm^-3;
    
    * p, and density of free holes in cm^-3;
    
    * net_charge, the net space charge in e/cm^3.
    """
    EF_minus_Ei = Evac_minus_Ei - Evac_minus_EF
    n = ni * np.exp(EF_minus_Ei / kT_in_eV)
    p = ni**2 / n
    return {'n':n, 'p':p, 'net_charge':(p - n + charge_from_dopants)}

def Evac_minus_EF_from_charge(Evac_minus_Ei, ni, charge_from_dopants, net_charge):
    """
    What value of (vacuum level minus fermi level) yields a local net
    charge equal to net_charge?
    
    See local_charge() for units and definitions related to inputs and
    outputs.
    
    Function is NOT vectorized. Inputs must be floats, not arrays. (This
    function is not called in inner loops, so speed doesn't matter.) 
    """
    # eh_charge is the charge from electrons and holes only
    eh_charge = net_charge - charge_from_dopants
    
    if eh_charge > 30 * ni:
        # Plenty of holes, negligible electrons
        p = eh_charge
        return Evac_minus_Ei + kT_in_eV * math.log(p / ni)
    if eh_charge < -30 * ni:
        # Plenty of electrons, negligible holes
        n = -eh_charge
        return Evac_minus_Ei - kT_in_eV * math.log(n / ni)
    
    # Starting here, we are in the situation where BOTH holes and electrons
    # need to be taken into account. Solve the simultaneous equations
    # p * n = ni**2 and p - n = eh_charge to get p and n.
        
    def solve_quadratic_equation(a,b,c):
        """ return larger solution to ax^2 + bx + c = 0 """
        delta = b**2 - 4 * a * c
        if delta < 0:
            raise ValueError("No real solution...that shouldn't happen!")
        return (-b + math.sqrt(delta)) / (2*a)

    if eh_charge > 0:
        # Slightly more holes than electrons
        p = solve_quadratic_equation(1, -eh_charge, -ni**2)
        return Evac_minus_Ei + kT_in_eV * math.log(p / ni)
    else:
        # Slightly more electrons than holes
        n = solve_quadratic_equation(1, eh_charge, -ni**2)
        return Evac_minus_Ei - kT_in_eV * math.log(n / ni)
    

def calc_core(points, eps, charge_from_dopants, Evac_minus_Ei, ni,
              tol=1e-5, max_iterations=inf, Evac_start=None, Evac_end=None):
    """
    Core routine for the calculation. Since it's a bit unweildy to input all
    these parameters by hand, you should normally use the wrapper
    calc_layer_stack() below.
    
    Inputs
    ------
    
    * points is a numpy list of coordinates, in nm, where we will find Evac.
      They must be in increasing order and equally spaced.
    
    * eps is a numpy list with the static dielectric constant at each point
      (unitless, i.e. epsilon / epsilon0)
    
    * charge_from_dopants is a numpy list with the net charge (in e/cm^3
      where e>0 is the elementary charge) from ionized donors or acceptors
      at each point. Normally one assumes all dopants are ionized, so it's
      equal to the doping for n-type, or negative the doping for p-type.
    
    * Evac_minus_Ei is a numpy list of the [positive] energy difference (in
      eV) between the vacuum level and the "intrinsic" fermi level at each
      point. ("intrinsic" means the fermi level at which p=n).
    
    * ni is a numpy list with the intrinsic electron concentration (in cm^-3)
      at each point. (Defined by p = n = ni in undoped equilibrium)
    
    * tol (short for tolerance) specifies the stopping point. A smaller
      number gives more accurate results. Each iteration step, we check
      whether Evac at any point moved by more than tol (in eV). If not, then
      terminate. Note: This does NOT mean that the answer will be within
      tol of the exact answer. Suggestion: Try 1e-4, 1e-5, 1e-6, etc. until
      the answer stops visibly changing.
    
    * max_iterations: How many iterations to do, before quitting even if the
      algorithm has not converged.
    
    * Evac_start is "vacuum energy minus fermi level in eV" at the first
      point. If it's left at the default value (None), we choose the value
      that makes it charge-neutral.
    
    * Evac_end is ditto for the end of the last layer.
    
    Method
    ------
    
    Since this is equilibrium, the fermi level is flat. We set it as the
    zero of energy.
    
    Start with Gauss's law:
    
    Evac'' = net_charge / epsilon
    
    (where '' is second derivative in space.)
    
    (Remember Evac = -electric potential + constant. The minus sign is
    because Evac is related to electron energy, and electrons have negative
    charge.)
    
    Using finite differences,
    (1/2) * dx^2 * Evac''[i] = (Evac[i+1] + Evac[i-1])/2 - Evac[i]
    
    Therefore, the MAIN EQUATION WE SOLVE:
    
    Evac[i] = (Evac[i+1] + Evac[i-1])/2 - (1/2) * dx^2 * net_charge[i] / epsilon
    
    ALGORITHM: The RHS at the previous time-step gives the LHS at the
    next time step. A little twist, which suppresses a numerical
    oscillation, is that net_charge[i] is inferred not from the Evac[i] at
    the last time step, but instead from the (Evac[i+1] + Evac[i-1])/2 at
    the last time step. The first and last values of Evac are kept fixed, see
    above.
        
    SEED: Start with the Evac profile wherein everything is charge-neutral.
    
    Output
    ------
    
    The final Evac (vacuum energy level) array (in eV). This is equivalent
    to minus the electric potential in V.
    """
    dx = points[1] - points[0]
    if max(np.diff(points)) > 1.001 * dx or min(np.diff(points)) < 0.999 * dx:
        raise ValueError('Error! points must be equally spaced!')
    if dx <= 0:
        raise ValueError('Error! points must be in increasing order!')
    
    num_points = len(points)
    
    # Seed for Evac
    seed_charge = np.zeros(num_points)
    Evac = [Evac_minus_EF_from_charge(Evac_minus_Ei[i], ni[i],
                                      charge_from_dopants[i], seed_charge[i])
                                          for i in range(num_points)]
    Evac = np.array(Evac)
    if Evac_start is not None:
        Evac[0] = Evac_start
    if Evac_end is not None:
        Evac[-1] = Evac_end

    ###### MAIN LOOP ######
    
    iters=0
    err=inf
    while err > tol and iters < max_iterations:
        iters += 1
        
        prev_Evac = Evac
        
        Evac = np.zeros(num_points)
        
        Evac[0] = prev_Evac[0]
        Evac[-1] = prev_Evac[-1]
        # Set Evac[i] = (prev_Evac[i-1] + prev_Evac[i+1])/2
        Evac[1:-1] = (prev_Evac[0:-2] + prev_Evac[2:])/2
        charge = local_charge(Evac_minus_Ei, ni, charge_from_dopants,
                                                         Evac)['net_charge']
        Evac[1:-1] -= 0.5 * dx**2 * charge[1:-1] / (eps[1:-1]
                                         * eps0_in_e_per_cm3_over_V_per_nm2)
        
        err = max(abs(prev_Evac - Evac))

        if False:
            # Optional: graph Evac a few times during the process to see
            # how it's going.
            if 5 * iters % max_iterations < 5:
                plt.figure()
                plt.plot(points, prev_Evac, points, Evac)
    if iters == max_iterations:
        print('Warning! Did not meet error tolerance. Evac changed by up to ('
                + '{:e}'.format(err) + ')eV in the last iteration.'  )
    else:
        print('Met convergence criterion after ' + str(iters)
               + ' iterations.')
    
    return Evac



############################################################################
############# MORE CONVENIENT INTERFACE / WRAPPERS #########################
############################################################################

class Material:
    """
    Semiconductor material with the following properties...
    
    NC = conduction-band effective density of states in cm^-3
    
    NV = valence-band effective density of states in cm^-3
    
    EG = Band gap in eV
    
    chi = electron affinity in eV (i.e. difference between conduction
          band and vacuum level)
    
    eps = static dielectric constant (epsilon / epsilon0)
    
    ni = intrinsic electron concentration in cm^-3 (defined by p = n = ni
    when the undoped material is in thermal equilibrium)
    
    Evac_minus_Ei is the [positive] energy difference (in eV) between the
        vacuum level and the "intrinsic" fermi level, i.e. the fermi level
        at which p=n.
    
    name = a string describing the material (for plot labels etc.)
    """

    def __init__(self, NC, NV, EG, chi, eps, name=''):
        self.NC = NC
        self.NV = NV
        self.EG = EG
        self.chi = chi
        self.eps = eps
        self.name = name
        
        # Sze equation (29), p21...
        self.ni = math.sqrt(self.NC * self.NV * math.exp(-self.EG / kT_in_eV))
        # Sze equation (27), p20...
        self.Evac_minus_Ei = (self.chi + 0.5 * self.EG
                              + 0.5 * kT_in_eV * math.log(self.NC / self.NV))

#Sze Appendix G

GaAs = Material(NC=4.7e17,
                NV=7.0e18,
                EG=1.42,
                chi=4.07,
                eps=12.9,
                name='GaAs')

Si = Material(NC=2.8e19,
              NV=2.65e19,
              EG=1.12,
              chi=4.05,
              eps=11.9,
              name='Si')

class Layer:
    """
    Layer of semiconductor with the following properties...
    
    matl = a material (an object with Material class)
    
    n_or_p = a string, either 'n' or 'p', for the doping polarity
    
    doping = density of dopants in cm^-3
    
    thickness = thickness of the layer in nm
    """
    def __init__(self, matl, n_or_p, doping, thickness):
        self.matl = matl
        self.n_or_p = n_or_p
        self.doping = doping
        self.thickness = thickness

def where_am_I(layers, distance_from_start):
    """
    distance_from_start is the distance from the start of layer 0.
    
    layers is a list of each layer; each element should be a Layer object.
    
    Return a dictionary {'current_layer':X, 'distance_into_layer':Y}.
    (Note: X is a Layer object, not an integer index.)
    """
    d = distance_from_start
    if distance_from_start < 0:
        raise ValueError('Point is outside all layers!')
    layer_index = 0
    while layer_index <= (len(layers) - 1):
        current_layer = layers[layer_index]
        if distance_from_start <= current_layer.thickness:
            return {'current_layer':current_layer,
                    'distance_into_layer':distance_from_start}
        else:
            distance_from_start -= current_layer.thickness
            layer_index += 1
    raise ValueError('Point is outside all layers! distance_from_start='
                       + str(d))


def calc_layer_stack(layers, num_points, tol=1e-5, max_iterations=inf,
                     Evac_start=None, Evac_end=None):
    """
    This is a wrapper around calc_core() that makes it more convenient to
    use. See example1(), example2(), etc. (below) for samples.
    
    Inputs
    ------
    
    * layers is a list of the "layers", where each "layer" is a Layer
      object.
    
    * num_points is the number of points at which to solve for Evac.
      (They will be equally spaced.)
    
    * tol, max_iterations, Evac_start, and Evac_end are defined the same as
      in calc_core() above.
    
    Outputs
    -------
    
    A dictionary with...
    
    * 'points', the 1d array of point coordinates (x=0 is the start of
    the 0'th layer.)
    
    * 'Evac', the 1d array of vacuum energy level in eV
    """
    total_thickness = sum(layer.thickness for layer in layers)
    points = np.linspace(0, total_thickness, num=num_points)
    # Note: layer_list is NOT the same as layers = [layer0, layer1, ...],
    # layer_list is [layer0, layer0, ... layer1, layer1, ... ], i.e. the
    # layer of each successive point.
    layer_list = [where_am_I(layers, pt)['current_layer']
                              for pt in points]
    matl_list = [layer.matl for layer in layer_list]
    eps = np.array([matl.eps for matl in matl_list])
    charge_from_dopants = np.zeros(num_points)
    for i in range(num_points):
        if layer_list[i].n_or_p == 'n':
            charge_from_dopants[i] = layer_list[i].doping
        elif layer_list[i].n_or_p == 'p':
            charge_from_dopants[i] = -layer_list[i].doping
        else:
            raise ValueError("n_or_p should be either 'n' or 'p'!")
    ni = np.array([matl.ni for matl in matl_list])
    Evac_minus_Ei = np.array([matl.Evac_minus_Ei for matl in matl_list])
    
    Evac = calc_core(points, eps, charge_from_dopants, Evac_minus_Ei, ni,
                           tol=tol, max_iterations=max_iterations,
                           Evac_start=Evac_start, Evac_end=Evac_end)
    return {'points':points, 'Evac':Evac}


def plot_bands(calc_layer_stack_output, layers):
    """
    calc_layer_stack_output is an output you would get from running
    calc_layer_stack(). layers is defined as in calc_layer_stack()
    """
    points = calc_layer_stack_output['points']
    Evac = calc_layer_stack_output['Evac']
    num_points = len(points)
    
    # Note: layer_list is NOT the same as layers = [layer0, layer1, ...],
    # layer_list is [layer0, layer0, ... layer1, layer1, ... ], i.e. the
    # layer of each successive point.
    layer_list = [where_am_I(layers, pt)['current_layer']
                              for pt in points]
    matl_list = [layer.matl for layer in layer_list]
    chi_list = [matl.chi for matl in matl_list]
    EG_list = [matl.EG for matl in matl_list]
    CB_list = [Evac[i] - chi_list[i] for i in range(num_points)]
    VB_list = [CB_list[i] - EG_list[i] for i in range(num_points)]
    EF_list = [0 for i in range(num_points)]
    
    plt.figure()
    
    plt.plot(points,CB_list,'k-', #conduction band: solid black line
             points,VB_list,'k-', #valence band: solid black line
             points,EF_list,'r--') #fermi level: dashed red line
    
    # Draw vertical lines at the boundaries of layers
    for i in range(len(layers)-1):
        plt.axvline(sum(layer.thickness for layer in layers[0:i+1]),color='k')
    
    # The title of the graph describes the stack
    # for example "1.3e18 n-Si / 4.5e16 p-Si / 3.2e17 n-Si"
    layer_name_string_list =  ['{:.1e}'.format(layer.doping) + ' '
                               + layer.n_or_p + '-' + layer.matl.name
                                                for layer in layers]
    plt.title(' / '.join(layer_name_string_list))
    plt.xlabel('Position (nm)')
    plt.ylabel('Electron energy (eV)')
    plt.xlim(0, sum(layer.thickness for layer in layers))

############################################################################
############################### EXAMPLES ###################################
############################################################################


def example1():
    """
    Example 1: Plot the equilibrium band diagram for an p / n junction
    """
    # doping density is in cm^-3; thickness is in nm.
    layer0 = Layer(matl=Si, n_or_p='p', doping=1e16, thickness=350)
    layer1 = Layer(matl=Si, n_or_p='n', doping=2e16, thickness=200)
    
    layers = [layer0, layer1]
    
    temp = calc_layer_stack(layers, num_points=100, tol=1e-6, max_iterations=inf)
    
    plot_bands(temp, layers)


def example2():
    """ Example 2: Plot the equilibrium band diagram for an n+ / n junction """
    # doping density is in cm^-3; thickness is in nm.
    layer0 = Layer(matl=GaAs, n_or_p='n', doping=1e17, thickness=100)
    layer1 = Layer(matl=GaAs, n_or_p='n', doping=1e15, thickness=450)
    
    layers = [layer0, layer1]
    
    temp = calc_layer_stack(layers, num_points=100, tol=1e-6, max_iterations=inf)
    
    plot_bands(temp, layers)


def example3():
    """
    Example 3: Plot the equilibrium band diagram for a BJT-like n+ / p / n / n+
    junction. Parameters based on example in Sze chapter 5 page 248.
    
    Note that num_points is a pretty large number (fine mesh). If you try a
    coarser mesh, the algorithm does not converge, but rather diverges with
    oscillations. (It's very sensitive when there are heavily-doped layers.)
    
    Remember that this simulation will NOT be quantitatively accurate,
    because I used a formula for n in terms of EF that is not valid at high
    concentration (it neglects band-filling and nonparabolicity).
    """
    # doping density is in cm^-3; thickness is in nm.
    layer0 = Layer(matl=Si, n_or_p='n', doping=1e20, thickness=150)
    layer1 = Layer(matl=Si, n_or_p='p', doping=1e18, thickness=100)
    layer2 = Layer(matl=Si, n_or_p='n', doping=3e16, thickness=150)
    layer3 = Layer(matl=Si, n_or_p='n', doping=1e20, thickness=50)
    
    layers = [layer0, layer1, layer2, layer3]
    
    temp = calc_layer_stack(layers, num_points=600, tol=1e-4, max_iterations=inf)
    
    plot_bands(temp, layers)


def example4():
    """
    Example 4: n-silicon with surface depletion (due to a gate or Schottky
    contact.)
    """
    layer0 = Layer(matl=Si, n_or_p='n', doping=1e15, thickness=1000.)
    
    layers = [layer0]
    
    temp = calc_layer_stack(layers, num_points=100, tol=1e-6,
                     max_iterations=inf, Evac_start=4.7)
        
    plot_bands(temp, layers)


def compare_to_depletion_approx(p_doping, n_doping, matl):
    """
    Compare my program to the full-depletion approximation for a p-n junction
    (wherein you calculate the potential profile by assuming n=p=0 in a
    certain region and that the material is charge-neutral outside that
    region).
    
    Ref: http://ecee.colorado.edu/~bart/book/pnelec.htm
    """
    # vacuum level on the p-side and n-side far from the junction
    Evac_lim_p = Evac_minus_EF_from_charge(matl.Evac_minus_Ei, matl.ni,
                                charge_from_dopants=-p_doping, net_charge=0)
    Evac_lim_n = Evac_minus_EF_from_charge(matl.Evac_minus_Ei, matl.ni,
                                charge_from_dopants=n_doping, net_charge=0)
    
    # built-in voltage in V
    built_in_voltage = Evac_lim_p - Evac_lim_n

    # w is the total depletion width in nm in the depletion approximation
    w = math.sqrt(2 * matl.eps * eps0_in_e_per_cm3_over_V_per_nm2
              * (1/n_doping + 1/p_doping) * built_in_voltage)
    
    # Here is the numerical calculation...
    p_layer_width = 1.5 * w
    n_layer_width = 1.5 * w
    layer0 = Layer(matl=matl, n_or_p='p', doping=p_doping, thickness=p_layer_width)
    layer1 = Layer(matl=matl, n_or_p='n', doping=n_doping, thickness=n_layer_width)
    layers = [layer0, layer1]
    temp = calc_layer_stack(layers, num_points=300, tol = 1e-6,
                     max_iterations=inf)
    Evac_numerical = temp['Evac']
    points = temp['points']
    
    # Back to the analytical, depletion-approximation calculation
    
    # xp and xn are depletion widths on n and p sides respectively
    xp = w * n_doping / (p_doping + n_doping)
    xn = w * p_doping / (p_doping + n_doping)
    
    # second derivative of Evac in the depletion region
    Evac_2nd_deriv_p = -p_doping / (matl.eps * eps0_in_e_per_cm3_over_V_per_nm2)
    Evac_2nd_deriv_n = n_doping / (matl.eps * eps0_in_e_per_cm3_over_V_per_nm2)
    
    # Coordinates for the start and end of the depletion region
    depletion_edge_p = p_layer_width - xp
    depletion_edge_n = p_layer_width + xn
    
    # Function giving the analytical value for Evac
    def Evac_analytical_fn(x):
        if x < depletion_edge_p:
            # Point is outside depletion region on p-side
            return Evac_lim_p
        if x > depletion_edge_n:
            # Point is outside depletion region on n-side
            return Evac_lim_n
        if x < p_layer_width:
            # Point is in depletion region on p-side
            return Evac_lim_p + 0.5 * Evac_2nd_deriv_p * (x - depletion_edge_p)**2
        else:
            # Point is in depletion region on n-side
            return Evac_lim_n + 0.5 * Evac_2nd_deriv_n * (x - depletion_edge_n)**2
   
    Evac_analytical = [Evac_analytical_fn(x) for x in points]
    plt.figure()
    plt.plot(points, Evac_analytical, 'b', points, Evac_numerical, 'r')
    layer_name_string_list =  ['{:.1e}'.format(layer.doping) + ' '
                               + layer.n_or_p + '-' + layer.matl.name
                                                for layer in layers]
    plt.title('Blue: Depletion approximation. Red: Numerical calculation.\n'
              + ' / '.join(layer_name_string_list))
    plt.xlabel('Position (nm)')
    plt.ylabel('Vacuum level (eV)')
    plt.xlim(0,p_layer_width + n_layer_width)
    # Draw vertical line at boundary
    plt.axvline(p_layer_width,color='k')

def example5():
    """
    Example 4: A p-n junction: Does the numerical calculation agree with the
    depletion approximation?
    """
    compare_to_depletion_approx(1e16, 2e16, Si)

############################################################################
########## SOME TESTS THAT HELPER-FUNCTIONS ARE CODED CORRECTLY ############
############################################################################


def local_charge_and_Ei_test():
    """ Test that Ei calculation is consistent with local_charge()"""
    print('p should equal n here...')
    print(local_charge(GaAs.Evac_minus_Ei, GaAs.ni, 0, GaAs.Evac_minus_Ei))

def Evac_minus_EF_from_charge_test():
    """
    Check that Evac_minus_EF_from_charge() is correct
    """
    matl = GaAs
    Evac_minus_Ei = matl.Evac_minus_Ei
    ni = matl.ni
    for doping_type in ['n','p']:
        for doping in [0, matl.ni, matl.ni*1e3]:
            charge_from_dopants = doping * (1 if doping_type=='n' else -1)
            for target_net_charge in set([-doping, 0, doping]):
                temp1 = Evac_minus_EF_from_charge(Evac_minus_Ei, ni,
                                                  charge_from_dopants,
                                                  target_net_charge)
                temp2 = local_charge(Evac_minus_Ei, ni, charge_from_dopants,
                                     temp1)
                print('\nDoping: ' '{:.3e}'.format(doping), doping_type,
                      ', Net charge goal:', '{:.3e}'.format(target_net_charge))
                print('n:', '{:.3e}'.format(temp2['n']),
                      '  p:', '{:.3e}'.format(temp2['p']),
                      '  net charge:', '{:.3e}'.format(temp2['net_charge']))

def where_am_I_test():
    """Test that where_am_I() is coded correctly"""
    # doping density should be in cm^-3; thickness should be in nm.
    layer0 = Layer(matl=GaAs, n_or_p='n', doping=1e18, thickness=100)
    layer1 = Layer(matl=Si, n_or_p='n', doping=1e16, thickness=50)
    layers = [layer0, layer1]
    print('The following should be True...')
    print(where_am_I(layers, 23)['current_layer'] is layer0)
    print(where_am_I(layers, 23)['distance_into_layer'] == 23)
    print(where_am_I(layers,123)['current_layer'] is layer1)
    print(where_am_I(layers,123)['distance_into_layer'] == 23)
