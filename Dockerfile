
FROM debian:latest

# installation les dépendances nécessaires pour Rust et curl
RUN apt-get update && \
    apt-get install -y curl build-essential

# Installation Rust avec rustup
# L'option `-y` répond automatiquement "oui" aux invites de `rustup`
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# l'ajout le chemin d'accès de cargo à PATH pour que les commandes Rust soient disponibles dans le conteneur
ENV PATH="/root/.cargo/bin:${PATH}"

# on copie le code source dans l'image
COPY . /usr/src/myapp

# changement du répertoire
WORKDIR /usr/src/myapp

# Construire de l'application peut etre fait à l'execution
#RUN cargo build --release

# on défini l'entrée par défaut pour exécuter les tests
ENTRYPOINT ["cargo", "test", "--release", "--"]

#pour l'execution sur ARM on utilise QEMU
#sudo apt-get install qemu qemu-user-static qemu-user binfmt-support
#puis on configure binfmt_misc pour supporter l'exécution de binaires ARM
#docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
#puis construction et execution 
#docker build -t moseiik-test .
#docker run --platform linux/arm64/v8 moseiik-test pas armv7

