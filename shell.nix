# save this as shell.nix
{pkgs ? import <nixpkgs> {}}:
pkgs.mkShell {
  packages = with pkgs; [
    (
      python3.withPackages
      (python-pkgs:
        with python-pkgs; [
          cvxpy
          numpy
          scipy
          sympy
        ])
    )
    gurobi
  ];
}
