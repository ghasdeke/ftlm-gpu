function [bonds, N, name, short] = geometry(geom, N_ring)
%FTLM.GEOMETRY  Nearest-neighbor bond list of a predefined cluster.
%   [BONDS, N, NAME, SHORT] = FTLM.GEOMETRY(GEOM) returns the bond list
%   BONDS (N_b x 2, 1-based site indices, i < j) of the cluster GEOM,
%   one of 'ico', 'cubo', 'cube', 'dodeca', 'icosid', 'ring'.  For
%   'ring', the number of sites is given as second argument N_RING.
%
%   The bond order is fixed (lexicographic in the vertex numbering of
%   the coordinate lists below) and identical in the Python package.

% ================================================================
% Copyright 2026 Shadan Ghassemi Tabrizi, Technische Universitaet Dresden,
% and Helmholtz-Zentrum Dresden-Rossendorf e.V.
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
% ================================================================

switch geom
    case 'ico'
        bonds = adjacency_icosahedron();
        N = 12;  name = 'Icosahedron';       short = 'ico';
    case 'cubo'
        bonds = adjacency_cuboctahedron();
        N = 12;  name = 'Cuboctahedron';     short = 'cubo';
    case 'cube'
        bonds = adjacency_cube();
        N = 8;   name = 'Cube';              short = 'cube';
    case 'dodeca'
        bonds = adjacency_dodecahedron();
        N = 20;  name = 'Dodecahedron';      short = 'dodeca';
    case 'icosid'
        bonds = adjacency_icosidodecahedron();
        N = 30;  name = 'Icosidodecahedron'; short = 'icosid';
    case 'ring'
        assert(nargin >= 2 && ~isempty(N_ring) && isnumeric(N_ring) && isscalar(N_ring) ...
               && N_ring == round(N_ring) && N_ring >= 3, ...
               'ftlm:geometry', 'geometry ''ring'' requires N_ring (integer >= 3).');
        N = N_ring;
        bonds = adjacency_ring(N);
        name  = sprintf('%d-Ring', N);
        short = sprintf('ring_%d', N);
    otherwise
        error('ftlm:geometry', ...
              'Unknown geometry: %s (allowed: ico, cubo, cube, dodeca, icosid, ring)', geom);
end
end

function bonds = adjacency_icosahedron()
%ADJACENCY_ICOSAHEDRON  30 edges of the icosahedron (12 vertices, z=5).
    phi = (1 + sqrt(5)) / 2;
    V = [0,1,phi;  0,1,-phi;  0,-1,phi;  0,-1,-phi;
         1,phi,0;  1,-phi,0; -1,phi,0;  -1,-phi,0;
         phi,0,1;  phi,0,-1; -phi,0,1;  -phi,0,-1];
    bonds = edges_at_distance(V, 2, 0.01);
    assert(size(bonds, 1) == 30);
end

function bonds = adjacency_cuboctahedron()
%ADJACENCY_CUBOCTAHEDRON  24 edges of the cuboctahedron (12 vertices, z=4).
    V = [ 1, 1, 0;  1,-1, 0; -1, 1, 0; -1,-1, 0;
          1, 0, 1;  1, 0,-1; -1, 0, 1; -1, 0,-1;
          0, 1, 1;  0, 1,-1;  0,-1, 1;  0,-1,-1];
    bonds = edges_at_distance(V, sqrt(2), 0.01);
    assert(size(bonds, 1) == 24);
end

function bonds = adjacency_cube()
%ADJACENCY_CUBE  12 edges of the cube (8 vertices, z=3).
    V = [-1,-1,-1; 1,-1,-1; -1,1,-1; 1,1,-1;
         -1,-1, 1; 1,-1, 1; -1,1, 1; 1,1, 1];
    bonds = edges_at_distance(V, 2, 0.01);
    assert(size(bonds, 1) == 12);
end

function bonds = adjacency_dodecahedron()
%ADJACENCY_DODECAHEDRON  30 edges of the dodecahedron (20 vertices, z=3).
    phi = (1 + sqrt(5)) / 2;
    V = [-1,-1,-1;  1,-1,-1; -1, 1,-1;  1, 1,-1;
         -1,-1, 1;  1,-1, 1; -1, 1, 1;  1, 1, 1;
          0, 1/phi, phi;   0,-1/phi, phi;   0, 1/phi,-phi;   0,-1/phi,-phi;
          1/phi, phi, 0;  -1/phi, phi, 0;   1/phi,-phi, 0;  -1/phi,-phi, 0;
          phi, 0, 1/phi;   phi, 0,-1/phi;  -phi, 0, 1/phi;  -phi, 0,-1/phi];
    bonds = edges_at_distance(V, 2 / phi, 0.1);
    assert(size(bonds, 1) == 30);
end

function bonds = adjacency_icosidodecahedron()
%ADJACENCY_ICOSIDODECAHEDRON  60 edges of the icosidodecahedron (30 vertices, z=4).
    phi   = (1 + sqrt(5)) / 2;
    polar = [0,0,phi; 0,0,-phi; phi,0,0; -phi,0,0; 0,phi,0; 0,-phi,0];
    sc    = (1 + phi) / 2;
    equat = [
        1/2, phi/2, sc;   -1/2, phi/2, sc;    1/2,-phi/2, sc;   -1/2,-phi/2, sc;
        1/2, phi/2,-sc;   -1/2, phi/2,-sc;    1/2,-phi/2,-sc;   -1/2,-phi/2,-sc;
        phi/2, sc, 1/2;   -phi/2, sc, 1/2;    phi/2,-sc, 1/2;   -phi/2,-sc, 1/2;
        phi/2, sc,-1/2;   -phi/2, sc,-1/2;    phi/2,-sc,-1/2;   -phi/2,-sc,-1/2;
        sc, 1/2, phi/2;   -sc, 1/2, phi/2;    sc,-1/2, phi/2;   -sc,-1/2, phi/2;
        sc, 1/2,-phi/2;   -sc, 1/2,-phi/2;    sc,-1/2,-phi/2;   -sc,-1/2,-phi/2];
    bonds = edges_at_distance([polar; equat], 1, 0.1);
    assert(size(bonds, 1) == 60);
end

function bonds = adjacency_ring(N)
%ADJACENCY_RING  N edges of a periodic 1D chain.
    bonds = [(1:N-1)', (2:N)'; N, 1];
end

function bonds = edges_at_distance(V, d, tol)
%EDGES_AT_DISTANCE  All vertex pairs i < j with |V_i - V_j| = d (+/- tol).
    n = size(V, 1);
    bonds = zeros(0, 2);
    for i = 1 : n
        for j = i+1 : n
            if abs(norm(V(i,:) - V(j,:)) - d) < tol
                bonds(end+1, :) = [i, j]; %#ok<AGROW>
            end
        end
    end
end
