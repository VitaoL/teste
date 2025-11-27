class Avaliacao {
    #id;
    #alunoNome;
    #professor;
    #nota;
    #comentario;
    #aulaId;

    constructor(id, alunoNome, professor, nota, comentario, aulaId = null) {
        if (nota < 1 || nota > 5) {
            throw new Error('A nota da avaliação deve ser um valor entre 1 e 5.');
        }

        this.#id = id;
        this.#alunoNome = alunoNome;
        this.#professor = professor;
        this.#nota = nota;
        this.#comentario = comentario;
        this.#aulaId = aulaId;
    }

    obterId() { return this.#id; }
    obterAlunoNome() { return this.#alunoNome; }
    obterProfessor() { return this.#professor; }
    obterNota() { return this.#nota; }
    obterComentario() { return this.#comentario; }
    obterAulaId() { return this.#aulaId; }
}

module.exports = Avaliacao;
