# This is a Nix flake for setting up a development environment for Torkius-VAD.
# It includes tooling and dependencies used in development.
#
# **If you are not a Nix user, you can ignore this file.**
#
# If you are curious to find out what this is, see <https://zero-to-nix.com/concepts/dev-env/>.

{
  description = "Flake for developing Torkius-VAD";

  inputs = {
    nixpkgs.url = "https://channels.nixos.org/nixos-25.05/nixexprs.tar.xz";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        inputsDeps = [
          # Avoid runtime errors in numpy-dependent scripts when running them
          # in the host env by using Nix's native package management for Python.
          # <https://gist.github.com/GuillaumeDesforges/7d66cf0f63038724acf06f17331c9280>
          (pkgs.python312.withPackages (
            python-pkgs: with python-pkgs; [
              numpy
              pandas
              matplotlib
            ]
          ))
        ];

        inputsTooling = [
          pkgs.uv
          pkgs.tree
          pkgs.pre-commit

          # Totally optional.
          pkgs.nodejs_24
        ];

        inputsLsp = [
          # Python.
          pkgs.pyright
          pkgs.ruff

          # Nix.
          pkgs.nixd
          pkgs.nixfmt-rfc-style

          # HTML.
          pkgs.htmx-lsp

          pkgs.prettierd
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
          ]
          ++ inputsDeps
          ++ inputsTooling
          ++ inputsLsp;

          shellHook = ''
            source ./.venv/bin/activate
          '';
        };

        # For compatibility with older versions of the `nix` binary.
        devShell = self.devShells.${system}.default;

        # Formatter to use with the `nix fmt` command.
        formatter = pkgs.nixfmt-tree;
      }
    );
}
