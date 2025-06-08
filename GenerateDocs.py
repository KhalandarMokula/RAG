from unstructured.partition.pdf import partition_pdf
from bs4 import BeautifulSoup
import re
from langchain.schema import Document

class GenerateDocs:
    def __init__(self, pdfPath):
        self.path = pdfPath
        self.name = pdfPath

    def ParsePDF(self):

        self.chunks = partition_pdf(
            filename=self.path,
            infer_table_structure=True,            # extract tables
            strategy="hi_res",                     # mandatory to infer tables
            extract_image_block_types=["Image", "Table"],   # Add 'Table' to list to extract image of tables
            # image_output_dir_path=output_path,   # if None, images and tables will saved in base64
            extract_image_block_to_payload=True,   # if true, will extract base64 for API usage
            chunking_strategy="by_title",          # or 'basic', by_title
            max_characters=10000,                  # defaults to 500
            combine_text_under_n_chars=2000,       # defaults to 0
            new_after_n_chars=6000,
        )

    def GetItems(self):
        self.resChunks = []
        self.resElms = []

        for chunk in self.chunks:
            elements = chunk.metadata.orig_elements
            resList=[]
            for elm in elements:
                # print(type(elm))
                # continue
                if 'Table' not in str(type(elm)):
                    #print(elm.to_dict())
                    elmDict = elm.to_dict()
                    text = [elmDict['text'] if 'text' in elmDict else '']
                    links = [elmDict['metadata']['links'] if 'links' in elmDict['metadata'] else []]
                    pageNumber = [elmDict['metadata']['page_number'] if 'page_number' in elmDict['metadata'] else []]
                    item = {"text": text, "links" : links, "pageNumber" : pageNumber}
                    resList.append(item)
                    self.resElms.append(item)
                else:
                    elmDict = elm.to_dict()
                    text = [elmDict['text'] if 'text' in elmDict else '']
                    tableHtml = [elmDict['metadata']['text_as_html'] if 'text_as_html' in elmDict['metadata'] else '']
                    links = [elmDict['metadata']['links'] if 'links' in elmDict['metadata'] else []]
                    pageNumber = [elmDict['metadata']['page_number'] if 'page_number' in elmDict['metadata'] else []]
                    item = {"tableHtml": tableHtml, "links" : links, "pageNumber" : pageNumber}
                    resList.append(item)
                    self.resElms.append(item)
            self.resChunks.append(resList)

        return self.resElms

    def clean_text(self, text):
        # Normalize whitespace and remove unwanted characters
        return re.sub(r'\s+', ' ', text.strip().replace('\xa0', ' '))

    def parse_table_to_paragraph(self, table_html):
        soup = BeautifulSoup(table_html, "html.parser")
        rows = soup.find_all("tr")
        lines = []

        for tr in rows:
            cols = tr.find_all(["td", "th"])
            col_texts = [self.clean_text(col.get_text()) for col in cols]

            if col_texts:
                line = " | ".join(col_texts)
                lines.append(line)

        paragraph = "\n".join(lines)
        return paragraph

    def GetDocs(self):
        self.ParsePDF()
        resElms = self.GetItems()
        completeDoc = ''
        docs = []
        smallDocs = []

        for item in resElms:
            if 'text' in item:
                completeDoc += item['text'][0]+ '\n'
            else:
                doc = Document(
                  page_content=completeDoc,
                  metadata = {"Policy": self.name}
                )
                docs.append(doc)
                #Extract Table data
                table = self.parse_table_to_paragraph(item['tableHtml'][0])
                nColumns = len(table.split('\n')[0].split('|'))
                columns = ['Important Questions', 'Answers', 'Why This Matters']
                
                columns4 = ['Common Medical Event' , 'Services You May Need' , 'Member out of pocket, Limitations, Exceptions', 'Other Important Information']

                rows = table.split('\n')

                for row in rows:
                    rowColsData = row.split('|')
                    if len(rowColsData) == 4:
                        metadata = {"event": rowColsData[0], "services": rowColsData[1]}
                        data = {columns4[0]: columns4[0], columns4[1]: rowColsData[1], columns4[2]: rowColsData[2], columns4[3]: rowColsData[3]}
                        pageContent = "\n".join(f"{key}: {value}" for key, value in data.items())
                    elif len(rowColsData) == 3:
                        metadata = {"event": rowColsData[0]}
                        data = {columns[0]: rowColsData[0], columns[1]: rowColsData[1], columns[2]: rowColsData[2]}
                        pageContent = "\n".join(f"{key}: {value}" for key, value in data.items())
                    else :
                        print(rowColsData)
                        #Examples data
                        if len(rowColsData) == 1 and rowColsData[0] == "Cost Sharing" :
                            #start new Doc
                            doc = Document(
                                page_content = '\n'.join(smallDocs),
                                metadata = {"Policy": self.name}
                            )
                            docs.append(doc)
                            continue
                        else:
                            x = rowColsData[0] + ':' +(rowColsData[1] if len(rowColsData) > 1 else '')
                            smallDocs.append(x)
                            continue

                    metadata['Policy'] = self.name
                    doc = Document(
                        page_content=pageContent,
                        metadata=metadata
                    )
                    docs.append(doc)
                    completeDoc = ''
        return docs


if __name__ == "__main__":
    import os
    directory_path = "Data"  # change this to your target folder

    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    print(files)
        
    allDocs = []        
    for filePath in files:
    
        genDocs = GenerateDocs("Data/" + filePath)
        docs = genDocs.GetDocs()
        allDocs.append(docs)

        DB_PATH = "chroma_db"
        vectorstore = Chroma.from_documents(
                documents=docs, 
                embedding=embeddings,
                persist_directory=DB_PATH
        )
        print("---------------------------------------")
        print(f"Ingestion complete. Vector store created at: {DB_PATH}")
        print(f"Total chunks stored: {vectorstore._collection.count()}")
        print("---------------------------------------")
