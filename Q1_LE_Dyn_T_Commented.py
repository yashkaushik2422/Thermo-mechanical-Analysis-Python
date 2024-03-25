import numpy as np

# The 2D quadrilateral element for Coupled Problems WS21/22
NoDim = 2
NoNodes = 4
# Each node carries three degrees of freedom:
#                   UX -> displacement in x-direction
#                   UY -> displacement in y-direction
#                   DT -> temperature increment T = T0 + DT
NoNodalDofs = 3
# For the temporal discretization we need for each node:
#                        h_I = [a_1, a_2, v_1, v_2, u_1, u_2, ddotT, dotT, T]
NoNodalHist = 9


# Material parameters are:
#                   E    -> Young's modulus
#                   nu   -> Poisson's ratio
#                   rho  -> density
#                   c_m  -> damping coefficient
#                   bx   -> gravity in x-direction
#                   by   -> gravity in y-direction
#                   c_a  -> thermal conductivity
#                   c_T  -> specific heat capacity
#                   r    -> volumetric heat source
#                   T0   -> stress free reference temperature
#                   c_Tu -> heat expansion coefficient
#                   cs   -> multiplicator on the nonlinear coupling term (0 -> off, 1-> on)

def Elmt_Init():
    '''
    The Elmt_Init() function is used to introduce the element to PyFEMP.
    '''
    NoElementDim = NoDim  # number of dimensions
    NoElementNodes = NoNodes  # number of nodes of this element
    NoElementHistory = NoNodes * NoNodalHist  # number of scalar history parameters
    ElementDofNames = ["UX", "UY", "DT"]  # list with string names for each dof
    # list with string name for material paramter
    ElementMaterialNames = ["E", "nu", "rho", "c_m", "bx", "by", "c_a", "c_T", "r", "T0", "c_Tu", "cs"]
    # list with string name for postprocessing
    ElementPostNames = ["UX", "UY", "T", "SigMises"]
    return NoElementDim, NoElementNodes, ElementDofNames, NoElementHistory, ElementMaterialNames, ElementPostNames


def Elmt_KS(XL, UL, Hn, Ht, Mat, dt):
    '''
    The Elmt_KS(XL, UL, Hn, Ht, Mat, dt) function returns the element vector and matrix.
    Input: 
            XL  = [x11, x12, x21, x22, ..., xn2]    -> np.vector of nodal coordinates for n-Nodes in 2d  
            UL  = [u11, u12, u21, u22, ..., un2]    -> np.vector of current nodal dofs for n-Nodes and 2 dofs at each
            Hn  = [h1, h2,... hm]                   -> np.vector for the previous history of length m (! do NOT write in here)
            Ht  = [h1, h2,... hm]                   -> np.vector for the updated history of length m (!write in here)
            Mat = [M1, M2, ... Mo]                  -> list of material parameter in the order defined in Elmt_Init()
            dt  = dt                                -> time step size
    '''
    # Initialize global matrices for element stiffness and residual
    r_e = np.zeros(NoNodes * NoNodalDofs)
    k_e = np.zeros((NoNodes * NoNodalDofs, NoNodes * NoNodalDofs))

    # Material properties
    Emod, nu, rho, c_m, bx, by, c_a, c_T, r, T0, c_Tu, cs = Mat
    b = np.array([bx, by])

    # XL: Initial nodal coordinates, UL: Initial displacements
    xI = np.array([[XL[0], XL[1]], [XL[2], XL[3]], [XL[4], XL[5]], [XL[6], XL[7]]])
    uI = np.array([[UL[0], UL[1]], [UL[3], UL[4]], [UL[6], UL[7]], [UL[9], UL[10]]])
    DTI = np.array([UL[2], UL[5], UL[8], UL[11]])

    # State variables at previous time step for Newmark time integration
    aIn = np.array([[Hn[0], Hn[1]], [Hn[2], Hn[3]], [Hn[4], Hn[5]], [Hn[6], Hn[7]]])
    vIn = np.array([[Hn[8], Hn[9]], [Hn[10], Hn[11]], [Hn[12], Hn[13]], [Hn[14], Hn[15]]])
    uIn = np.array([[Hn[16], Hn[17]], [Hn[18], Hn[19]], [Hn[20], Hn[21]], [Hn[22], Hn[23]]])
    ddDTIn = np.array([Hn[24 + 0], Hn[24 + 1], Hn[24 + 2], Hn[24 + 3]])
    dDTIn = np.array([Hn[24 + 4], Hn[24 + 5], Hn[24 + 6], Hn[24 + 7]])
    DTIn = np.array([Hn[24 + 8], Hn[24 + 9], Hn[24 + 10], Hn[24 + 11]])

    # Calculate acceleration, velocity, and second derivative of temperature at nodes by Newmark time integration
    gamma = 1.0 / 2.0
    beta = 1.0 / 4.0
    aI = 1.0 / (beta * dt * dt) * (uI - uIn - dt * vIn - dt * dt * (1.0 / 2.0 - beta) * aIn)
    vI = gamma / (beta * dt) * (uI - uIn) + (1.0 - gamma / beta) * vIn + dt * (1.0 - gamma / (2.0 * beta)) * aIn
    ddDTI = 1.0 / (beta * dt * dt) * (DTI - DTIn - dt * dDTIn - dt * dt * (1.0 / 2.0 - beta) * ddDTIn)
    dDTI = gamma / (beta * dt) * (DTI - DTIn) + (1.0 - gamma / beta) * dDTIn + dt * (
                1.0 - gamma / (2.0 * beta)) * ddDTIn

    # Update state variables at nodes
    # Ht: Updated state variables at nodes
    [[Ht[0], Ht[1]], [Ht[2], Ht[3]], [Ht[4], Ht[5]], [Ht[6], Ht[7]]] = aI
    [[Ht[8], Ht[9]], [Ht[10], Ht[11]], [Ht[12], Ht[13]], [Ht[14], Ht[15]]] = vI
    [[Ht[16], Ht[17]], [Ht[18], Ht[19]], [Ht[20], Ht[21]], [Ht[22], Ht[23]]] = uI
    [Ht[24 + 0], Hn[24 + 1], Ht[24 + 2], Ht[24 + 3]] = ddDTI
    [Ht[24 + 4], Hn[24 + 5], Ht[24 + 6], Ht[24 + 7]] = dDTI
    [Ht[24 + 8], Ht[24 + 9], Ht[24 + 10], Ht[24 + 11]] = DTI

    # Define Gauss points and weights for numerical integration
    # EGP: Gauss points in the natural coordinate system
    aa = 1 / np.sqrt(3)
    EGP = np.array([[-aa, -aa, 1], [aa, -aa, 1], [aa, aa, 1], [-aa, aa, 1]])
    NoInt = len(EGP)

    # Loop over Gauss points for numerical integration
    for GP in range(NoInt):
        xi, eta, wgp = EGP[GP]

        # Calculate shape functions and their derivatives
        SHP = 1 / 4 * np.array(
            [(1.0 - xi) * (1.0 - eta), (1.0 + xi) * (1.0 - eta), (1.0 + xi) * (1.0 + eta), (1.0 - xi) * (1.0 + eta)])
        SHP_dxi = 1 / 4 * np.array([[-(1.0 - eta), -(1.0 - xi)],
                                    [(1.0 - eta), -(1.0 + xi)],
                                    [(1.0 + eta), (1.0 + xi)],
                                    [-(1.0 + eta), (1.0 - xi)]
                                    ], dtype=np.float64)

        # Calculate Jacobian matrix, its determinant, and inverse
        Jac = np.zeros((2, 2))
        for I in range(NoNodes):
            for i in range(2):
                for j in range(2):
                    Jac[i, j] += SHP_dxi[I, j] * xI[I, i]

        detJ = np.linalg.det(Jac)
        Jinv = np.linalg.inv(Jac)

        # Calculate gradient of shape functions
        SHP_dx = np.zeros((NoNodes, 2))
        for I in range(NoNodes):
            for i in range(2):
                for j in range(2):
                    SHP_dx[I, i] += Jinv[j, i] * SHP_dxi[I, j]

                    # Calculate acceleration for Gauss Point
        a = np.zeros(2)
        for I in range(NoNodes):
            for i in range(2):
                a[i] += SHP[I] * aI[I, i]

        # Calculate second derivative of temperature for Gauss Point
        dT = 0.0
        for I in range(NoNodes):
            dT += SHP[I] * dDTI[I]

        # Calculate the total temperature at the Gauss point
        DT = 0.0
        for I in range(NoNodes):
            DT += SHP[I] * DTI[I]

        # Calculate strains at Gaus Point
        eps = np.zeros(6)
        for I in range(NoNodes):
            # B-matrix for node I
            BI = np.array([[SHP_dx[I, 0], 0],
                           [0, SHP_dx[I, 1]],
                           [0, 0],
                           [SHP_dx[I, 1], SHP_dx[I, 0]],
                           [0, 0],
                           [0, 0]
                           ])
            for i in range(6):
                for j in range(2):
                    eps[i] += BI[i, j] * uI[I, j]

        # compute Voigt identity
        II = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        # compute strain rate
        deps = np.zeros(6)
        for I in range(NoNodes):
            # compute B-matrix for this node I
            BI = np.array([[SHP_dx[I, 0], 0],
                           [0, SHP_dx[I, 1]],
                           [0, 0],
                           [SHP_dx[I, 1], SHP_dx[I, 0]],
                           [0, 0],
                           [0, 0]
                           ])
            for i in range(6):
                for j in range(2):
                    deps[i] += BI[i, j] * vI[I, j]

        # Identity matrix for stress-strain relationship
        iden = np.zeros((2, 2))
        for i in range(2):
            iden[i, i] = 1.0

        # lam: Lame's first parameter, mue: Shear modulus, kappa: Bulk modulus
        lam = (Emod * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mue = Emod / (2.0 * (1.0 + nu))
        kappa = Emod / (3.0 * (1.0 - 2.0 * nu))

        # Calculate material stiffness matrix and damping matrix

        Cmat = np.array([
            [lam + 2 * mue, lam, lam, 0, 0, 0],
            [lam, lam + 2 * mue, lam, 0, 0, 0],
            [lam, lam, lam + 2 * mue, 0, 0, 0],
            [0, 0, 0, mue, 0, 0],
            [0, 0, 0, 0, mue, 0],
            [0, 0, 0, 0, 0, mue]
        ], dtype=np.float64)

        Dmat = c_m * np.array([
            [2 / 3, -1 / 3, -1 / 3, 0, 0, 0],
            [-1 / 3, 2 / 3, -1 / 3, 0, 0, 0],
            [-1 / 3, -1 / 3, 2 / 3, 0, 0, 0],
            [0, 0, 0, 1 / 2, 0, 0],
            [0, 0, 0, 0, 1 / 2, 0],
            [0, 0, 0, 0, 0, 1 / 2]
        ], dtype=np.float64)

        # Calculate stress vector at Gauss points
        sig = np.zeros(6)
        for i in range(6):
            for m in range(6):
                sig[i] += Cmat[i, m] * eps[m] + Dmat[i, m] * deps[m]

            sig[i] += - 3.0 * c_Tu * kappa * DT * II[i]

        # Calculate temperature gradient
        grad_T = np.zeros(2)
        for I in range(NoNodes):
            # temperatute gradient at Gauss Points
            bI = np.array([SHP_dx[I, 0], SHP_dx[I, 1]])

            for i in range(2):
                grad_T[i] += bI[i] * DTI[I]

        # Calculate heat flux vector
        q = - c_a * grad_T

        # Calculate element vector and matrix
        for I in range(NoNodes):
            # select shape function at node I
            NI = SHP[I]
            # compute B-matrix for this node I
            BI = np.array([[SHP_dx[I, 0], 0],
                           [0, SHP_dx[I, 1]],
                           [0, 0],
                           [SHP_dx[I, 1], SHP_dx[I, 0]],
                           [0, 0],
                           [0, 0]
                           ])

            bI = np.array([SHP_dx[I, 0], SHP_dx[I, 1]])

            # #Balance of Momentum
            for k in range(2):
                r_e[I * 3 + k] += rho * (a[k] - b[k]) * NI * detJ * wgp

                for i in range(6):
                    r_e[I * 3 + k] += sig[i] * BI[i, k] * detJ * wgp

                    # #Balance of Energy
            for i in range(2):
                r_e[I * 3 + 2] += -q[i] * bI[i] * detJ * wgp

            r_e[I * 3 + 2] += -rho * r * NI * detJ * wgp
            r_e[I * 3 + 2] += rho * c_T * dT * NI * detJ * wgp

            # Compute the contribution to the internal forces and stiffness matrix from the thermal strain and temperature gradient

            for i in range(2):
                for j in range(2):
                    # # Contribution to internal forces due to thermal strain 
                    r_e[I * 3 + 2] += - Dmat[i, j] * deps[i] * deps[j] * NI * detJ * wgp * cs

            for J in range(NoNodes):
                # Shape function at node J
                NJ = SHP[J]
                # Shape function gradients at node J
                BJ = np.array([[SHP_dx[J, 0], 0],
                               [0, SHP_dx[J, 1]],
                               [0, 0],
                               [SHP_dx[J, 1], SHP_dx[J, 0]],
                               [0, 0],
                               [0, 0]
                               ])
                # Nodal gradients at node J
                bJ = np.array([SHP_dx[J, 0], SHP_dx[J, 1]])

                for k in range(2):
                    for l in range(2):
                        # Contribution to stiffness matrix from thermal strain
                        k_e[I * 3 + k, J * 3 + l] += rho / (beta * dt * dt) * NI * NJ * iden[k, l] * detJ * wgp

                        for i in range(6):
                            for m in range(6):
                                # Contribution to stiffness matrix from material constitutive model and thermal strain
                                k_e[I * 3 + k, J * 3 + l] += (Cmat[i, m] + gamma / (beta * dt) * Dmat[i, m]) \
                                                             * BJ[m, l] * BI[i, k] * detJ * wgp

                # Contribution to stiffness matrix from thermal stress and temperature gradient
                for i in range(2):
                    k_e[I * 3 + 2, J * 3 + 2] += c_a * bJ[i] * bI[i] * detJ * wgp

                # Contribution to stiffness matrix from thermal stress and temperature change
                k_e[I * 3 + 2, J * 3 + 2] += rho * c_T * NI * NJ * (gamma / (beta * dt)) * detJ * wgp

                # Contribution to stiffness matrix from thermal stress and thermal strain
                for i in range(2):
                    for k in range(6):
                        k_e[I * 3 + i, J * 3 + 2] += - 3.0 * c_Tu * kappa * NJ * II[k] * BI[k, i] * detJ * wgp

                        # Contribution to stiffness matrix from thermal stress and temperature gradient
                for k in range(2):
                    for i in range(2):
                        for j in range(2):
                            k_e[I * 3 + 2, J * 3 + k] += - ((2.0 * gamma) / (beta * dt)) * Dmat[i, j] * deps[i] * BJ[
                                j, k] * NI * detJ * wgp * cs

    return r_e, k_e


def Elmt_Post(XL, UL, Hn, Ht, Mat, dt, PostName):
    '''
    The Elmt_Post(XL, UL, Hn, Ht, Mat, dt, PostName) function returns a vector,
    containing a scalar for each node.
    '''

    # initialize return - the post function always returns a vector 
    # containing one scalar per node of the element
    r_post = np.zeros(4)

    # read in material parameter
    Emod, nu, rho, c_m, bx, by, c_a, c_T, r, T0, c_Tu, cs = Mat
    b = np.array([bx, by])

    # restructure input to fit our notation
    xI = np.array([[XL[0], XL[1]], [XL[2], XL[3]], [XL[4], XL[5]], [XL[6], XL[7]]])
    uI = np.array([[UL[0], UL[1]], [UL[3], UL[4]], [UL[6], UL[7]], [UL[9], UL[10]]])
    DTI = np.array([UL[2], UL[5], UL[8], UL[11]])

    # read histroy - Newmark time integration
    aIn = np.array([[Hn[0], Hn[1]], [Hn[2], Hn[3]], [Hn[4], Hn[5]], [Hn[6], Hn[7]]])
    vIn = np.array([[Hn[8], Hn[9]], [Hn[10], Hn[11]], [Hn[12], Hn[13]], [Hn[14], Hn[15]]])
    uIn = np.array([[Hn[16], Hn[17]], [Hn[18], Hn[19]], [Hn[20], Hn[21]], [Hn[22], Hn[23]]])
    ddDTIn = np.array([Hn[24 + 0], Hn[24 + 1], Hn[24 + 2], Hn[24 + 3]])
    dDTIn = np.array([Hn[24 + 4], Hn[24 + 5], Hn[24 + 6], Hn[24 + 7]])
    DTIn = np.array([Hn[24 + 8], Hn[24 + 9], Hn[24 + 10], Hn[24 + 11]])

    # compute current acceleration and velocity - Newmark time integration
    gamma = 1.0 / 2.0
    beta = 1.0 / 4.0
    aI = 1.0 / (beta * dt * dt) * (uI - uIn - dt * vIn - dt * dt * (1.0 / 2.0 - beta) * aIn)
    vI = gamma / (beta * dt) * (uI - uIn) + (1.0 - gamma / beta) * vIn + dt * (1.0 - gamma / (2.0 * beta)) * aIn
    ddDTI = 1.0 / (beta * dt * dt) * (DTI - DTIn - dt * dDTIn - dt * dt * (1.0 / 2.0 - beta) * ddDTIn)
    dDTI = gamma / (beta * dt) * (DTI - DTIn) + (1.0 - gamma / beta) * dDTIn + dt * (
                1.0 - gamma / (2.0 * beta)) * ddDTIn

    # write histroy - Newmark time integration
    [[Ht[0], Ht[1]], [Ht[2], Ht[3]], [Ht[4], Ht[5]], [Ht[6], Ht[7]]] = aI
    [[Ht[8], Ht[9]], [Ht[10], Ht[11]], [Ht[12], Ht[13]], [Ht[14], Ht[15]]] = vI
    [[Ht[16], Ht[17]], [Ht[18], Ht[19]], [Ht[20], Ht[21]], [Ht[22], Ht[23]]] = uI
    [Ht[24 + 0], Hn[24 + 1], Ht[24 + 2], Ht[24 + 3]] = ddDTI
    [Ht[24 + 4], Hn[24 + 5], Ht[24 + 6], Ht[24 + 7]] = dDTI
    [Ht[24 + 8], Ht[24 + 9], Ht[24 + 10], Ht[24 + 11]] = DTI

    # provide integration points
    aa = 1 / np.sqrt(3)
    EGP = np.array([[-aa, -aa, 1], [aa, -aa, 1], [aa, aa, 1], [-aa, aa, 1]])
    NoInt = len(EGP)

    # start integration Loop
    for GP in range(NoInt):
        xi, eta, wgp = EGP[GP]

        # evaluate shape functions at this gp
        SHP = 1 / 4 * np.array(
            [(1.0 - xi) * (1.0 - eta), (1.0 + xi) * (1.0 - eta), (1.0 + xi) * (1.0 + eta), (1.0 - xi) * (1.0 + eta)])
        SHP_dxi = 1 / 4 * np.array([[-(1.0 - eta), -(1.0 - xi)],
                                    [(1.0 - eta), -(1.0 + xi)],
                                    [(1.0 + eta), (1.0 + xi)],
                                    [-(1.0 + eta), (1.0 - xi)]
                                    ], dtype=np.float64)

        # compute Jacobin matrix
        Jac = np.zeros((2, 2))
        for I in range(NoNodes):
            for i in range(2):
                for j in range(2):
                    Jac[i, j] += SHP_dxi[I, j] * xI[I, i]

        # compute Jacobi- determinant and inverse
        detJ = np.linalg.det(Jac)
        Jinv = np.linalg.inv(Jac)

        # compute gradient shape functions
        SHP_dx = np.zeros((NoNodes, 2))
        for I in range(NoNodes):
            for i in range(2):
                for j in range(2):
                    SHP_dx[I, i] += Jinv[j, i] * SHP_dxi[I, j]

                    # compute acceleration at gp
        a = np.zeros(2)
        for I in range(NoNodes):
            for i in range(2):
                a[i] += SHP[I] * aI[I, i]

        # compute rate of temperature
        dT = 0.0
        for I in range(NoNodes):
            dT += SHP[I] * dDTI[I]

        # compute temperature relative to T0
        DT = 0.0
        for I in range(NoNodes):
            DT += SHP[I] * DTI[I]

        # compute strains
        eps = np.zeros(6)
        for I in range(NoNodes):
            # compute B-matrix for this node I
            BI = np.array([[SHP_dx[I, 0], 0],
                           [0, SHP_dx[I, 1]],
                           [0, 0],
                           [SHP_dx[I, 1], SHP_dx[I, 0]],
                           [0, 0],
                           [0, 0]
                           ])
            for i in range(6):
                for j in range(2):
                    eps[i] += BI[i, j] * uI[I, j]

        # compute Voigt identity
        II = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        # compute strain rate
        deps = np.zeros(6)
        for I in range(NoNodes):
            # compute B-matrix for this node I
            BI = np.array([[SHP_dx[I, 0], 0],
                           [0, SHP_dx[I, 1]],
                           [0, 0],
                           [SHP_dx[I, 1], SHP_dx[I, 0]],
                           [0, 0],
                           [0, 0]
                           ])
            for i in range(6):
                for j in range(2):
                    deps[i] += BI[i, j] * vI[I, j]

        # compute identity tensor
        iden = np.zeros((2, 2))
        for i in range(2):
            iden[i, i] = 1.0

        # form constitutive tensor
        lam = (Emod * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mue = Emod / (2.0 * (1.0 + nu))
        kappa = Emod / (3.0 * (1.0 - 2.0 * nu))

        Cmat = np.array([
            [lam + 2 * mue, lam, lam, 0, 0, 0],
            [lam, lam + 2 * mue, lam, 0, 0, 0],
            [lam, lam, lam + 2 * mue, 0, 0, 0],
            [0, 0, 0, mue, 0, 0],
            [0, 0, 0, 0, mue, 0],
            [0, 0, 0, 0, 0, mue]
        ], dtype=np.float64)

        Dmat = c_m * np.array([
            [2 / 3, -1 / 3, -1 / 3, 0, 0, 0],
            [-1 / 3, 2 / 3, -1 / 3, 0, 0, 0],
            [-1 / 3, -1 / 3, 2 / 3, 0, 0, 0],
            [0, 0, 0, 1 / 2, 0, 0],
            [0, 0, 0, 0, 1 / 2, 0],
            [0, 0, 0, 0, 0, 1 / 2]
        ], dtype=np.float64)

        # compute stresses
        sig = np.zeros(6)
        for i in range(6):
            for m in range(6):
                sig[i] += Cmat[i, m] * eps[m] + Dmat[i, m] * deps[m]

            sig[i] += - 3.0 * c_Tu * kappa * DT * II[i]

        # compute temperature gradient
        grad_T = np.zeros(2)
        for I in range(NoNodes):
            # compute b-vector for this node I
            bI = np.array([SHP_dx[I, 0], SHP_dx[I, 1]])

            for i in range(2):
                grad_T[i] += bI[i] * DTI[I]

        # compute heat flux
        q = - c_a * grad_T

        # compute vonMises stresses
        sig_vm = 0.0

        sig_vm = np.sqrt( \
            sig[0] ** 2 + sig[1] ** 2 + sig[2] ** 2 \
            - sig[0] * sig[1] - sig[0] * sig[2] - sig[1] * sig[2] \
            + 3 * (sig[3] ** 2 + sig[4] ** 2 + sig[5] ** 2) \
            )

        if (PostName == "SigMises"):
            r_post += sig_vm * SHP

    # based on the string PostName different output is returned
    if (PostName == "UX"):
        r_post = np.array([UL[0], UL[3], UL[6], UL[9]])
        return r_post
    elif (PostName == "UY"):
        r_post = np.array([UL[1], UL[4], UL[7], UL[10]])
        return r_post
    elif (PostName == "T"):
        r_post = np.array([T0 + UL[2], T0 + UL[5], T0 + UL[8], T0 + UL[11]])
        return r_post
    elif (PostName == "SigMises"):
        return r_post
    else:
        print("Waring: PostName " + PostName + " not defined!")
        return np.array([0.0, 0.0, 0.0, 0.0])


## This is a sanity check for the element

# define dummy input
XL = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
UL = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
Hn = np.zeros(4 * 3 * 3)
Ht = np.zeros(4 * 3 * 3)
Mat = [2100, 0.3, 1.0, 3.0, 0.1, 2.0, 10, 5.0, 0, 293.15, 0 * 10e-5, 0]
dt = 1
# call the elemnt with this dummy input
re, ke = Elmt_KS(XL, UL, Hn, Ht, Mat, dt)
# check the resulting vector / matrix
np.set_printoptions(linewidth=1000)
print('r_e :')
print(re)
print('k_e :')
print(ke)
