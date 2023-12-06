%% 1D Galerkin with implicit backward Euler

% Building the elemental mass and stiffness matrices 

% Mapping from x to xi
% Number of nodes, x left boundary, x right boundary, f(x)
N = 11;
xl = 0;
xr = 1;
f = @(x) sin(pi*x);

% Creating a uniform grid and connectivity map
Ne = N-1;
h = (xr-xl)/(N-1);
x = zeros(N);
iee = zeros(Ne,2);

for i = 1:Ne
    x(i) = xl+(i-1)*h;
    iee(i,1) = i;
    iee(i,2) = i+1; 
end

x(N) = xr;

% Defining parent grid basis functions and derivatives with the bounds
% -1<=xi<=1
phi1 = @(xi) (1-xi)/2;
phi2 = @(xi) (1+xi)/2;
dphi1 = -1/2;
dphi2 = 1/2;
dx = h/2;
dxi = 2/h;

% Initialization of K(stiffness matrix), F, and M, as well as the local
% matrices
K = zeros(N,N);
M = zeros(N,N);
F = zeros(N,1);
klocal = zeros(2,2);
flocal = zeros(2,1);
mlocal = [2,1;1,2];

for k = 1:Ne
    %% Attempting to solve with quadrature
    flocal(1) = dx*(f(0)*phi1(-1)+f(1)*phi1(1));
    flocal(2) = dx*(f(0)*phi2(-1)+f(1)*phi2(1));
    klocal(1,1) = (dxi*dphi1)*(dxi*dphi1)*dx;
    klocal(1,2) = (dxi*dphi1)*(dxi*dphi2)*dx;
    klocal(2,1) = (dxi*dphi2)*(dxi*dphi1)*dx;
    klocal(2,2) = (dxi*dphi2)*(dxi*dphi2)*dx;
    % Finite element assembly
    for l = 1:2
        global_node1 = iee(k,l);
        F(global_node1) = F(global_node1) + flocal(l);
        for m = 1:2
            global_node2 = iee(k,m);
            K(global_node1, global_node2) = K(global_node1, global_node2)+klocal(l,m);
            M(global_node1, global_node2) = M(global_node1, global_node2)+mlocal(l,l);
        end
    end
end


%% 1D Finite Element Heat Equation Addition with backward Euler
% Time step
dt = 1/551;
T0 = 0;
Tf = 1;
nt = (Tf-T0)/dt;
ctime = T0;

% Defining fhat function, flocal as fl, Fn which serves as F, and u
fhat = @(x,t) (pi^2-1)*exp(-t)*sin(pi*x);
fl = zeros(2,1);
Fn = zeros(N,1);
u = exp(-0)*sin(pi*x);
for n = 1:nt
    ctime = T0 + (n*dt);
    % Building the time-dependent RHS vector
    for k = 1:Ne
        % Attempting to solve with quadrature
        %fl(1) = dx*(fhat(0,ctime)*phi1(-1)+fhat(1,ctime)*phi1(1));
        %fl(2) = dx*(fhat(0,ctime)*phi2(-1)+fhat(1,ctime)*phi2(1));
        fl(1) = dx*(fhat(x(iee(k,1)),ctime)*phi1(-1)+fhat(x(iee(k,2)),ctime)*phi1(1));
        fl(2) = dx*(fhat(x(iee(k,1)),ctime)*phi2(-1)+fhat(x(iee(k,2)),ctime)*phi2(1));
        % Finite Element Assembly
        for l = 1:2
            global_node1 = iee(k,l);
            Fn(global_node1) = Fn(global_node1)+fl(l);
        end
    end
    B = (1/dt)*M+K;
    u = (1/dt)*pinv(B)*M*u+pinv(B)*F;
end

figure;
subplot(1,2,1);
plot(x,u,'b');
title('Calculated Solution at Final Time')
xlabel('x');
ylabel('u');

usol = exp(-nt)*sin(pi*x);
subplot(1,2,2);
plot(x,usol, 'r--');
title('Analytic Solution at Final Time')
xlabel('x');
ylabel('u');


