FROM debian:latest

# Mise à jour et installation des dépendances nécessaires pour Rust, curl, wget et unzip
RUN apt-get update && \
    apt-get install -y curl build-essential wget unzip

# Installation Rust avec rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Ajout du chemin d'accès de cargo à PATH
ENV PATH="/root/.cargo/bin:${PATH}"

# Copie du code source dans l'image
COPY . /usr/src/myapp

# Changement du répertoire de travail
WORKDIR /usr/src/myapp

# Téléchargement et décompression de l'archive d'images
RUN wget "https://filesender.renater.fr/download.php?token=178558c6-7155-4dca-9ecf-76cbebeb422e&files_ids=33679270" -O images.zip && \
    unzip images.zip -d /usr/src/myapp/images

# Construction de l'application (peut être fait à l'exécution)
#RUN cargo build --release

# Définition de l'entrée par défaut pour exécuter les tests
ENTRYPOINT ["cargo", "test", "--release", "--"]

#pour l'execution sur ARM on utilise QEMU
#sudo apt-get install qemu qemu-user-static qemu-user binfmt-support
#puis on configure binfmt_misc pour supporter l'exécution de binaires ARM
#docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
#puis construction et execution 
#docker build -t moseiik-test .
#docker run --platform linux/arm64/v8 moseiik-test pas armv7

