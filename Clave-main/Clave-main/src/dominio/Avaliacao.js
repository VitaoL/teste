class Avaliacao {
    #id;
    #aulaId;
    #alunoId;
    #professorId;
    #nota;
    #comentario;

    constructor(id, aulaId, alunoId, professorId, nota, comentario) {
        this.#id = id;
        this.#aulaId = aulaId;
        this.#alunoId = alunoId;
        this.#professorId = professorId;
        this.#nota = nota;
        this.#comentario = comentario;
    }

    obterId() { return this.#id; }
    obterAulaId() { return this.#aulaId; }
    obterAlunoId() { return this.#alunoId; }
    obterProfessorId() { return this.#professorId; }
    obterNota() { return this.#nota; }
    obterComentario() { return this.#comentario; }

    toJSON() {
        return {
            id: this.#id,
            nota: this.#nota,
            comentario: this.#comentario,
            aulaId: this.#aulaId
        };
    }
}

module.exports = Avaliacao;