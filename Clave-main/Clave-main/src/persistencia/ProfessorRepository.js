const Database = require('./database');
const Professor = require('../dominio/Professor');

class ProfessorRepository {

    // CREATE - Salvar Professor com Transação
    async salvar(professor) {
        const client = await Database.getPool().connect();
        
        try {
            await client.query('BEGIN');

            // 1. Inserir Professor (Atualizado com foto e vídeo)
            const sqlProf = `
                INSERT INTO professor (nome, email, senha_hash, biografia, valor_base_aula, link_video_demo, foto_url) 
                VALUES ($1, $2, $3, $4, $5, $6, $7) 
                RETURNING id
            `;
            
            const valuesProf = [
                professor.obterNome(),
                professor.obterEmail(),
                professor.obterSenhaHash(),
                // Usando toJSON().bio caso o getter direto não exista, ou acessando direto se implementou
                professor.toJSON().bio || "", 
                professor.obterValorBaseAula(),
                professor.obterLinkVideo() || null, // Novo campo
                professor.obterFotoUrl() || null    // Novo campo
            ];
            
            const resProf = await client.query(sqlProf, valuesProf);
            const professorId = resProf.rows[0].id;

            // 2. Vincular Instrumentos
            for (const nomeInstrumento of professor.obterInstrumentos()) {
                let resInst = await client.query('SELECT id FROM instrumento WHERE nome = $1', [nomeInstrumento]);
                let instrumentoId;
                
                if (resInst.rows.length > 0) {
                    instrumentoId = resInst.rows[0].id;
                } else {
                    const insertInst = await client.query('INSERT INTO instrumento (nome) VALUES ($1) RETURNING id', [nomeInstrumento]);
                    instrumentoId = insertInst.rows[0].id;
                }

                await client.query(
                    'INSERT INTO professor_instrumento (professor_id, instrumento_id) VALUES ($1, $2) ON CONFLICT DO NOTHING',
                    [professorId, instrumentoId]
                );
            }

            // 3. Vincular Estilos
            for (const nomeEstilo of professor.obterEstilos()) {
                let resEst = await client.query('SELECT id FROM estilo_musical WHERE nome = $1', [nomeEstilo]);
                let estiloId;

                if (resEst.rows.length > 0) {
                    estiloId = resEst.rows[0].id;
                } else {
                    const insertEst = await client.query('INSERT INTO estilo_musical (nome) VALUES ($1) RETURNING id', [nomeEstilo]);
                    estiloId = insertEst.rows[0].id;
                }

                await client.query(
                    'INSERT INTO professor_estilo (professor_id, estilo_id) VALUES ($1, $2) ON CONFLICT DO NOTHING',
                    [professorId, estiloId]
                );
            }

            await client.query('COMMIT');
            return professorId;

        } catch (error) {
            await client.query('ROLLBACK');
            throw error;
        } finally {
            client.release();
        }
    }

    // READ - Buscar Todos
    async buscarTodos() {
        const pool = Database.getPool();
        const sql = `
            SELECT 
                p.*,
                ARRAY_REMOVE(ARRAY_AGG(DISTINCT i.nome), NULL) as instrumentos,
                ARRAY_REMOVE(ARRAY_AGG(DISTINCT em.nome), NULL) as estilos
            FROM professor p
            LEFT JOIN professor_instrumento pi ON p.id = pi.professor_id
            LEFT JOIN instrumento i ON pi.instrumento_id = i.id
            LEFT JOIN professor_estilo pe ON p.id = pe.professor_id
            LEFT JOIN estilo_musical em ON pe.estilo_id = em.id
            GROUP BY p.id
            ORDER BY p.nome
        `;
        const { rows } = await pool.query(sql);

        return rows.map(row => new Professor(
            row.id, 
            row.nome, 
            row.email, 
            row.senha_hash, 
            row.biografia, 
            Number(row.valor_base_aula), 
            row.instrumentos || [], 
            row.estilos || [],
            row.link_video_demo, // Passando o vídeo para o construtor
            row.foto_url         // Passando a foto para o construtor
        ));
    }

    // READ - Buscar por ID
    async buscarPorId(id) {
        const pool = Database.getPool();
        const sql = `
            SELECT 
                p.*,
                ARRAY_REMOVE(ARRAY_AGG(DISTINCT i.nome), NULL) as instrumentos,
                ARRAY_REMOVE(ARRAY_AGG(DISTINCT em.nome), NULL) as estilos
            FROM professor p
            LEFT JOIN professor_instrumento pi ON p.id = pi.professor_id
            LEFT JOIN instrumento i ON pi.instrumento_id = i.id
            LEFT JOIN professor_estilo pe ON p.id = pe.professor_id
            LEFT JOIN estilo_musical em ON pe.estilo_id = em.id
            WHERE p.id = $1
            GROUP BY p.id
        `;
        const { rows } = await pool.query(sql, [id]);
        
        if (rows.length === 0) return null;
        const row = rows[0];

        return new Professor(
            row.id, 
            row.nome, 
            row.email, 
            row.senha_hash, 
            row.biografia, 
            Number(row.valor_base_aula), 
            row.instrumentos, 
            row.estilos,
            row.link_video_demo, // Passando o vídeo
            row.foto_url         // Passando a foto
        );
    }
    
    async buscarPorEmail(email) {
        const pool = Database.getPool();
        const sql = 'SELECT * FROM professor WHERE email = $1';
        const { rows } = await pool.query(sql, [email]);
        
        if (rows.length === 0) return null;
        const row = rows[0];
        
        // No login simples, passamos apenas os dados básicos
        // Se precisar de instrumentos no login, teria que fazer JOIN aqui também
        return new Professor(
            row.id, 
            row.nome, 
            row.email, 
            row.senha_hash, 
            row.biografia, 
            Number(row.valor_base_aula),
            [], // Instrumentos vazio
            [], // Estilos vazio
            row.link_video_demo,
            row.foto_url
        );
    }
}

module.exports = ProfessorRepository;