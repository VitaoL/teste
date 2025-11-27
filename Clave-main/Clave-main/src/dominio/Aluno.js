class Aluno {
    #id;
    #nome;
    #email;
    #telefone;
    #aulasAgendadas;

    constructor(id, nome, email, telefone = null) {
        this.#id = id;
        this.#nome = nome;
        this.#email = email;
        this.#telefone = telefone;
        this.#aulasAgendadas = [];
    }

    adicionarAula(aula) {
        this.#aulasAgendadas.push(aula);
    }

    obterId() { return this.#id; }
    obterNome() { return this.#nome; }
    obterEmail() { return this.#email; }
    obterTelefone() { return this.#telefone; }
    obterAulasAgendadas() { return this.#aulasAgendadas; }

    alterarNome(novoNome) {
        this.#nome = novoNome;
    }

    alterarTelefone(novoTelefone) {
        this.#telefone = novoTelefone;
    }
}

module.exports = Aluno;
