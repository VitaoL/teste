class Professor {
    #id;
    #nome;
    #email;
    #instrumento;
    #precoHora;
    #avaliacoes;

    constructor(id, nome, email, instrumento, precoHora) {
        this.#id = id;
        this.#nome = nome;
        this.#email = email;
        this.#instrumento = instrumento;
        this.#precoHora = precoHora;
        this.#avaliacoes = [];
    }

    adicionarAvaliacao(avaliacao) {
        this.#avaliacoes.push(avaliacao);
    }

    calcularNotaMedia() {
        if (this.#avaliacoes.length === 0) {
            return 0;
        }
        const soma = this.#avaliacoes.reduce((total, avaliacao) => total + avaliacao.obterNota(), 0);
        return soma / this.#avaliacoes.length;
    }

    alterarNome(novoNome) {
        this.#nome = novoNome;
    }

    obterId() { return this.#id; }
    obterNome() { return this.#nome; }
    obterEmail() { return this.#email; }
    obterInstrumento() { return this.#instrumento; }
    obterPrecoHora() { return this.#precoHora; }
    obterAvaliacoes() { return this.#avaliacoes; }
}

module.exports = Professor;
