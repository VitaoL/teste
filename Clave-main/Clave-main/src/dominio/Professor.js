class Professor {
    // Atributos privados
    #id;
    #nome;
    #email;
    #senhaHash;
    #biografia;
    #valorBaseAula;
    #instrumentos;
    #estilos;
    #linkVideo; // Novo campo
    #fotoUrl;   // Novo campo

    // Construtor atualizado para receber video e foto
    constructor(id, nome, email, senhaHash, biografia, valorBaseAula, instrumentos = [], estilos = [], linkVideo = null, fotoUrl = null) {
        this.#id = id;
        this.#nome = nome;
        this.#email = email;
        this.#senhaHash = senhaHash;
        this.#biografia = biografia;
        this.#valorBaseAula = valorBaseAula;
        this.#instrumentos = instrumentos;
        this.#estilos = estilos;
        this.#linkVideo = linkVideo;
        this.#fotoUrl = fotoUrl;
    }

    // --- Métodos de Negócio ---
    adicionarAulaNaAgenda(aula) {
        // Lógica futura de agenda
    }

    // --- Métodos de Acesso (Getters) ---
    obterId() { return this.#id; }
    obterNome() { return this.#nome; }
    obterEmail() { return this.#email; }
    obterSenhaHash() { return this.#senhaHash; }
    obterBiografia() { return this.#biografia; }
    obterValorBaseAula() { return this.#valorBaseAula; }
    obterInstrumentos() { return this.#instrumentos; }
    obterEstilos() { return this.#estilos; }
    obterLinkVideo() { return this.#linkVideo; }
    obterFotoUrl() { return this.#fotoUrl; }

    // --- Serialização para o Frontend ---
    // Transforma os dados privados em um objeto JSON público
    toJSON() {
        return {
            id: this.#id,
            name: this.#nome,
            email: this.#email,
            bio: this.#biografia,
            price: Number(this.#valorBaseAula),
            instruments: this.#instrumentos,
            styles: this.#estilos,
            // Campos visuais para o novo layout
            link_video_demo: this.#linkVideo, 
            image: this.#fotoUrl || `https://via.placeholder.com/300x200?text=${encodeURIComponent(this.#nome)}`,
            
            // Valores padrão para compatibilidade com o layout de cards
            rating: 5.0, 
            reviews: 0
        };
    }
}

module.exports = Professor;